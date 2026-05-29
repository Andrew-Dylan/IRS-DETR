# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

IRS-DETR is a modified RT-DETR object detector built on a fork of Ultralytics YOLOv8 (v8.0.201). The core innovation is replacing standard up/down-sampling in the feature pyramid neck with **Haar wavelet-based pooling and unpooling** (`WaveletPool` / `WaveletUnPool`), combined with **SHSA (Single-Head Self-Attention)** blocks in the backbone. The project serves as an experimental testbed for ablation studies on backbone and neck components.

## Dev environment

- Python conda environment: `rt` (interpreter at `D:/miniconda3/envs/rt/python.exe`)
- Run all scripts from repo root with `PYTHONPATH` set to the project root

## Common scripts

```bash
# Profile a YAML config (prints FLOPs, params, layer info)
python main_profile.py

# Check backbone output shapes for all supported backbones
python check_backbone.py

# Train (edit train.py first to set your YAML, data path, epochs, batch, etc.)
python train.py

# Validate (edit val.py first to set model .pt path and data YAML)
python val.py

# Detect/inference (edit detect.py first to set model and source)
python detect.py
```

All `train.py`, `val.py`, and `detect.py` import from the local `ultralytics/` package and require manual in-file editing of model path, data path, and other parameters before running.

## Architecture

### IRS-DETR model (primary)

Defined in [`Exp-model/IRS-DETR.yaml`](Exp-model/IRS-DETR.yaml):

**Backbone** (custom CSP-style):
```
Conv → Conv → C2f → Conv → C2f → Conv → C2f_SHSA → Conv → C2f_SHSA (×3)
```
The `C2f_SHSA` blocks replace standard C2f bottleneck blocks with Single-Head Self-Attention for global context modeling. Used in deeper layers of the backbone where feature maps are small enough that attention is affordable.

**Neck** (wavelet-based cross-scale fusion):
```
AIFI encoder → WaveletUnPool → concat with backbone skip → RepC3 →
WaveletUnPool → concat with backbone skip → RepC3 →
WaveletPool   → concat with previous    → RepC3 →
WaveletPool   → concat with previous    → RepC3
```
`WaveletUnPool` uses Haar wavelet filters in a transposed convolution for learnable-free 2× upsampling that preserves frequency content (LL, LH, HL, HH subbands). `WaveletPool` uses the same filters in a strided convolution for anti-aliased 2× downsampling. This replaces `nn.Upsample(nearest)` / `Conv(stride=2)` used in the baseline RT-DETR-R18.

**Head**: `AIFI` (Attention-based Intra-scale Feature Interaction) applies a transformer encoder to the coarsest feature map, then feeds three scale features into `RTDETRDecoder` (Deformable Transformer Decoder) for final bounding box prediction.

### Baseline RT-DETR-R18

Defined in [`Exp-model/RT-DETR-R18.yaml`](Exp-model/RT-DETR-R18.yaml) — standard RT-DETR using ResNet-18 backbone with nearest-neighbor upsampling and stride-2 conv downsampling in the neck. Used as the comparison baseline.

### Ablation configs

- `EFAModule.yaml` — C2f_SHSA backbone + standard up/down-sampling neck (tests SHSA contribution)
- `WaveSample.yaml` — standard ResNet backbone + wavelet pooling/unpooling neck (tests wavelet contribution)
- `MKPFusion.yaml` — alternative fusion module experiment

### Comparison backbones

YAMLs in `Exp-model/comparison-backbone/` test different backbone architectures (EfficientViT, FasterNet, MobileNetV4, RepViT, RMT, SwinTransformer) with the standard RT-DETR neck.

## Key code layout

- `ultralytics/` — the core library (fork of ultralytics/ultralytics v8.0.201, AGPL-3.0)
  - `nn/tasks.py` — model parsing from YAML and `BaseModel` / `RTDETRDetectionModel` classes
  - `nn/modules/` — standard YOLOv8 modules (Conv, C2f, RepC3, head, transformer)
  - `nn/extra_modules/` — ~96 custom module files (blocks, attention variants, fusion modules, etc.). `__init__.py` controls which are imported. All imports use wildcards so adding a class to the right file makes it available in YAML configs
  - `nn/backbone/` — custom backbone implementations (EfficientViT, FasterNet, Swin, etc.)
  - `models/rtdetr/` — RTDETR model class, trainer, predictor, validator
  - `engine/` — training loop, validation, prediction, export
  - `data/` — dataloaders and augmentations
  - `cfg/` — default configs and CLI entrypoint (`yolo` command)
- `Exp-model/` — YAML model definition files for experiments
- `dataset/` — dataset configuration YAMLs (ISDD single-class ship detection)

## Model config YAML format

The YAML files use a list-based DSL parsed by `ultralytics/nn/tasks.py`. Each layer is `[from_index, repeat_count, module_name, args]`. `from_index` of `-1` means the previous layer. Lists in `from_index` define concatenation (multiple inputs).

## Dataset format

Standard YOLO format. Dataset YAML (e.g., `dataset/ISDD.yaml`) specifies:
```yaml
path: /dataset/root
train: ./images/train
val: ./images/val
test: ./images/test
nc: 1
names: ['ship']
```

## Adding a new module

1. Implement the PyTorch module in `ultralytics/nn/extra_modules/` (or `modules/`)
2. If a new file, add a wildcard import in `ultralytics/nn/extra_modules/__init__.py`
3. Use the class name directly in a YAML config — no registration step needed

## Notes

- The project is not a git repository
- RT-DETR training notes: `F.grid_sample` doesn't support `deterministic=True`; AMP training can produce NaN and errors during bipartite matching
- The YAML parsing uses `-1` for previous layer, and lists like `[[6, 4], 1, Concat, [1]]` for concatenation
- Wildcard imports (`*`) are used everywhere — never remove seemingly unused imports without checking they're referenced via YAML configs
