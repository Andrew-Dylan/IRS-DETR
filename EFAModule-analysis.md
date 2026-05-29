# EFAModule 改动分析

## 目标

将 `Exp-model\ablation-exp-model\EFAModule.yaml` 中的 `C2f_SHSA` 替换为 `EFAModule`。

## EFAModule 是什么

`EFAModule` 目前不存在于代码库中。按照命名推断，它应该是 `C2f` + `ExternalAttention` 的组合——与 `C2f_SHSA`（C2f + Single-Head Self-Attention）平行的消融对照模块，用 External Attention 替代 SHSA。

现有基础组件：
- `ExternalAttention`（已存在，`block.py:159`）——O(N×S) 复杂度的外部注意力，**forward 内部已有残差** (`F.relu(x + x_attn)`)
- `SHSABlock_FFN`（已存在，`block.py:98`）——可复用
- `Conv2d_BN`、`Residual`（已存在）

## 需要修改的文件

### 1. `ultralytics/nn/extra_modules/block.py` — 新增 `EFABlock` 和 `EFAModule`

`C2f_SHSA` 的结构（`block.py:150-153`）：

```python
class C2f_SHSA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SHSABlock(self.c) for _ in range(n))
```

`SHSABlock` 内部结构（`block.py:140-148`）：

```python
# x → [dw-conv + residual] → [SHSA + residual] → [FFN + residual]
class SHSABlock(nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))
```

新增代码放在 `C2f_SHSA` 之后、`MFPM chain` 注释之前（约 `block.py:154`），如下：

```python
class EFABlock(nn.Module):
    """External Attention block: dw-conv → ExternalAttention → FFN, each with residual."""
    def __init__(self, dim):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        self.ea = ExternalAttention(dim)            # 内部已有残差，不再套 Residual
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn(self.ea(self.conv(x)))


class EFAModule(C2f):
    """C2f wrapper with EFABlock bottlenecks instead of Bottleneck."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EFABlock(self.c) for _ in range(n))
```

### 2. `ultralytics/nn/tasks.py` — 添加 `EFAModule` 到模型解析器

需要在 `parse_model` 中给 `EFAModule` 与 `C2f_SHSA` 相同的处理，**修改两处**：

**位置 1：行 ~768**，在 `C2f_SHSA` 后面加 `EFAModule`：

```python
# 修改前
                C2f_SHSA,
                ):
# 修改后
                C2f_SHSA,
                EFAModule,
                ):
```

**位置 2：行 ~777**，在 `C2f_SHSA` 后面加 `EFAModule`：

```python
# 修改前
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3, C2f_SHSA):
# 修改后
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3, C2f_SHSA, EFAModule):
```

### 3. `Exp-model/ablation-exp-model/EFAModule.yaml` — 替换模块名

将 `C2f_SHSA` 全部替换为 `EFAModule`（共 2 处，第 6 行和第 8 行）：

```yaml
# 修改前
  - [-1, 1, C2f_SHSA, [384]]
  ...
  - [-1, 3, C2f_SHSA, [384]]
# 修改后
  - [-1, 1, EFAModule, [384]]
  ...
  - [-1, 3, EFAModule, [384]]
```

## 不需要修改的地方

- `ultralytics/nn/extra_modules/__init__.py` — 已是 `from .block import *`，自动导出新类
- `ExternalAttention` — 已存在，无需改动
- 其余 YAML 文件 — 不改动

## 验证方法

```bash
cd E:\IRS-DETR
python main_profile.py  # 确认 [PASS] EFAModule.yaml
```

或者单独测试：

```python
from ultralytics import RTDETR
model = RTDETR('Exp-model/ablation-exp-model/EFAModule.yaml')
model.model.eval()
import torch
with torch.no_grad():
    y = model.model(torch.randn(1, 3, 640, 640))
print('OK')
```
