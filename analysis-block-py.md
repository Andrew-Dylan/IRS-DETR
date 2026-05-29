# `extra_modules/block.py` 依赖分析

## 概要

## 扫描范围

- `Exp-model/` 下全部 11 个 YAML 配置文件
- `ultralytics/` 下全部 Python 文件（`nn/extra_modules/`、`nn/modules/`、`nn/backbone/`、`models/`、`engine/` 等）
- `train.py`、`val.py`、`detect.py`、`main_profile.py`、`check_backbone.py`
- 交叉导入链：`from .block import X` / `from ultralytics.nn.extra_modules import X` / `from . import X`（经 `__init__.py` 通配符）

## 全局 YAML 用到的模块

| 模块 | 来源文件 | 引用 YAML |
|---|---|---|
| `C2f_SHSA` | **extra_modules/block.py** | IRS-DETR, EFAModule |
| `WaveletPool` | **extra_modules/block.py** | IRS-DETR, WaveSample |
| `WaveletUnPool` | **extra_modules/block.py** | IRS-DETR, WaveSample |
| `MFPM` | **extra_modules/block.py** | MKPFusion |
| `AIFI` | modules/transformer.py | 全部 11 个 |
| `BasicBlock` | modules/block.py | RT-DETR-R18, WaveSample, MKPFusion, rtdetr-r18 |
| `Blocks` | modules/block.py | 同上 |
| `C2f` | modules/block.py | IRS-DETR, EFAModule |
| `ConvNormLayer` | modules/block.py | RT-DETR-R18, WaveSample, MKPFusion, rtdetr-r18 |
| `RepC3` | modules/block.py | 全部 11 个 |
| `Concat` | modules/conv.py | 全部 11 个 |
| `Conv` | modules/conv.py | 全部 11 个 |
| `RTDETRDecoder` | modules/head.py | 全部 11 个 |
| `EfficientViT_M0` | backbone/efficientViT.py | rtdetr-EfficientViT |
| `fasternet_t0` | backbone/fasternet.py | rtdetr-fasternet |
| `MobileNetV4ConvSmall` | backbone/mobilenetv4.py | rtdetr-mobilenetv4 |
| `repvit_m0_9` | backbone/repvit.py | rtdetr-repvit |
| `RMT_T` | backbone/rmt.py | RT-DETR-R18, rtdetr-RMT |
| `SwinTransformer_Tiny` | backbone/SwinTransformer.py | rtdetr-SwinTiny |
| `nn.MaxPool2d` | PyTorch | RT-DETR-R18, WaveSample, MKPFusion, rtdetr-r18 |
| `nn.Upsample` | PyTorch | RT-DETR-R18 等 8 个 YAML |

> 加粗行来自 `extra_modules/block.py`。

## 必须保留的 13 个类（按依赖链分组）

### 链 1：Wavelet 小波采样（2 个类）— WaveSample / IRS-DETR

```
WaveletPool    (line 5671)  — Haar 小波下采样，stride=2 抗混叠
WaveletUnPool  (line 5691)  — Haar 小波上采样，转置卷积恢复空间分辨率
```

这两个类自包含，仅依赖 `torch` / `numpy` / `torch.nn.functional`，无内部交叉引用。

### 链 2：C2f_SHSA 自注意力模块（7 个类）— EFAModule / IRS-DETR

```
C2f_SHSA (line 7417)                     — YAML 直接引用，替换标准 C2f
├── SHSABlock (line 7397)                — bottleneck 单元
│   ├── Residual (line 7294)             — 残差包装器（skip connection + 主路径）
│   ├── Conv2d_BN (line 7270)            — Conv2d + BatchNorm 组合层
│   ├── SHSA (line 7321)                 — Single-Head Self-Attention 核心
│   │   └── SHSA_GroupNorm (line 7302)   — 内置 GroupNorm（通道数自适应）
│   └── SHSABlock_FFN (line 7310)        — 前馈网络（Conv1x1 → Conv3x3 → Conv1x1）
```

SHSA 替代标准多头自注意力（MHSA），仅用一个注意力头 + GroupNorm，在深层 backbone 特征图（尺寸小、通道多）上降低计算量。

### 链 3：MFPM 多频金字塔融合（4 个类）— MKPFusion

```
MFPM (line 13777)                               — YAML 直接引用，多频金字塔混合模块
├── EA_MF (line 13755)                           — 外部注意力 + 多频融合
│   ├── ExternalAttention (line 13557)           — 双记忆单元外部注意力（M_k, M_v）
│   └── MultiFrequencyChannelAttention (line 13673) — DCT 多频通道注意力
```

## 交叉文件引用验证

- 无任何 `.py` 文件通过 `from .block import X` 显式导入 `block.py` 的类
- 无任何 `.py` 文件通过 `from ultralytics.nn.extra_modules import X` 导入 `block.py` 的类
- 无任何 `.py` 文件通过 `from . import X`（经 `__init__.py` 通配符链）间接导入 `block.py` 的类
- 其他 `extra_modules/` 下的文件各自通过独立 `__init__.py` 或直接模块导入，不依赖 `block.py`

YAML 中类的加载路径唯一：

```
train.py → RTDETR(model.yaml)
         → tasks.py: BaseModel.__init__ 解析 YAML
         → extra_modules/__init__.py: from .block import *
         → globals() 查找类名并实例化
```

## 可移除的 651 个类（按功能分类）

| 功能系列 | 示例 | 约计数 |
|---|---|---|
| DCNv2/v3/v4 可变形卷积 | `DCNv2`, `DCNV3_YOLO`, `DCNV4_YOLO`, 所有 Bottleneck/C3/C2f 变体 | 30 |
| iRMB 倒残差移动块 | `iRMB`, `iRMB_Cascaded`, `iRMB_DRB`, `iRMB_SWC`, 所有 C3/C2f/BottleNeck 变体 | 20 |
| Faster Block | `Faster_Block`, `Partial_conv3`, `Faster_Block_EMA`, 所有 C3/C2f/BottleNeck 变体 | 30 |
| DBB/DRB 多样化重参数化 | `DilatedReparamBlock`, `Bottleneck_DRB`, `Bottleneck_DBB`, 各变体 | 25 |
| RFAConv 感受野注意力卷积 | `Bottleneck_RFAConv`, `RFAConv`, `RFCAConv`, `RFCBAMConv`, 各变体 | 20 |
| AKConv 可变核卷积 | `AKConv`, 各变体 | 10 |
| VSS 视觉状态空间 | `Bottleneck_VSS`, `C3_VSS`, `C2f_VSS`, `C3_LVMB` 等 | 10 |
| DySample / CARAFE 上采样 | `DySample`, `CARAFE`, `HWD` | 3 |
| GOLDYOLO | `GOLDYOLO_Attention`, `top_Block`, `TopBasicLayer`, `AdvPoolFusion` | 4 |
| ContextGuided | `ContextGuidedBlock`, `ContextGuidedBlock_Down`, 各变体 | 8 |
| HSFPN | `ChannelAttention_HSFPN`, `ELA_HSFPN`, `CA_HSFPN`, `CAA_HSFPN` | 4 |
| Ortho 正交变换 | `GramSchmidtTransform`, `Attention_Ortho`, `BasicBlock_Ortho`, 各变体 | 10 |
| RepNCSP 重参数化跨阶段部分 | `RepNCSP`, `DBBNCSP`, `OREPANCSP`, `RepNCSPELAN4`, 各变体 | 15 |
| 其他单点模块 | `PKIModule`, `ERM`, `Zoom_cat`, `ScalSeq`, `SDI`, `FGlo` 等 | ~450 |

## 移除后影响

移除未使用类后各入口脚本行为：

| 脚本 | 状态 | 原因 |
|---|---|---|
| `train.py` | 正常运行 | YAML → `tasks.py` → `extra_modules/__init__.py` 依赖链完整 |
| `val.py` | 正常运行 | 同上 |
| `detect.py` | 正常运行 | 同上 |
| `main_profile.py` | 正常运行 | 同上 |
| `check_backbone.py` | 正常运行 | 使用 `backbone/` 下独立文件，不依赖 `block.py` 移除部分 |

## 操作建议

采用**注释折叠**方式保留代码，便于按需恢复：

```python
# ================================================================
# KEPT: 被 Exp-model YAML 引用
# ================================================================
class WaveletPool(nn.Module):
    ...

# ================================================================
# REMOVED: 无 YAML 或 Python 文件引用
# ================================================================
if False:
    class ADown(nn.Module):
        ...

    class AKConv(nn.Module):
        ...
```

要点：
- 文件顶部 `import` 保持不变（未使用的 import 不影响运行，且某些被保留类间接依赖）
- `extra_modules/__init__.py` 中 `from .block import *` 保持不变，注释掉的类不会导出
- 预计文件从 ~14000 行缩减至 ~400 行（减少约 97%），但完整历史保留在文件中
