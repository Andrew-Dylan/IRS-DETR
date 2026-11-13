# IRS-DETR: Efficient Infrared Ship Detection Transformer for Complex Sea–Air Backgrounds

## 📖 Introduction

Infrared ship detection in maritime surveillance often suffers from complex sea-air backgrounds, heavy sea clutter, strong waves, and small targets with weak textures. Standard downsampling and conventional attention mechanisms either lose tiny targets or introduce high computational overhead.

**IRS-DETR** is a lightweight real-time Transformer detector based on the RT-DETR framework designed to achieve high detection accuracy while maintaining real-time inference speed in complex maritime environments.

---

## 📂 Project Structure

```text
IRS-DETR/
├── EXP-Model/                     # Model architecture YAML definitions
│   ├── IRS-DETR.yaml              # Proposed full IRS-DETR model
│   ├── RT-DETR.yaml               # Baseline RT-DETR model
│   └── Ablation-exp-model/        # Ablation study configurations
│       ├── EFAModule.yaml         # Ablation: EFAModule backbone only
│       └── WaveletSample.yaml     # Ablation: Wavelet sampling only
├── dataset/                       # Dataset configuration files
│   └── ISDD.yaml                  # Dataset configuration for ISDD
├── ultralytics/                   # Core engine & neural network definitions
│   ├── nn/
│   │   ├── extra_modules/         # Innovative blocks (EFAModule, WaveletPool, MFPM, etc.)
│   │   └── tasks.py               # Model parser & registry
│   └── ...
├── train_pipeline.py              # Automated multi-model training pipeline
├── main_profile.py                # Model profiling script (Parameters, GFLOPs, Fusing)
├── val.py                         # Model evaluation & testing script
├── detect.py                      # Inference & visualization script
├── requirements.txt               # Environment dependencies
└── setup.py                       # Project setup & packaging
```

---

## 🛠️ Environment & Installation

### Requirements
- **OS:** Linux / Windows
- **Python:** $\ge$ 3.8 (Recommended: Python 3.10 / 3.11)
- **PyTorch:** $\ge$ 2.0.0
- **TorchVision:** $\ge$ 0.15.0
- **CUDA:** 11.8 / 12.1 or newer

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Andrew-Dylan/IRS-DETR.git
   cd IRS-DETR
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```
   *or*
   ```bash
   python setup.py develop
   ```

---

## 📊 Dataset Preparation

This project uses the **ISDD (Infrared Ship Detection Dataset)**:
- Dataset Source: [https://github.com/yaqihan-9898/ISDD](https://github.com/yaqihan-9898/ISDD)

Organize your dataset directory as follows:
```text
/path/to/ISDD/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Update the root path in `dataset/ISDD.yaml`:
```yaml
path: /path/to/ISDD  # Replace with your actual dataset directory
train: ./images/train
val: ./images/val
test: ./images/test

nc: 1
names: ['ship']
```

---

## 🚀 Usage

### 1. Model Profiling (FLOPs & Parameters)
To inspect the layer structure, parameter counts, and GFLOPs across all model configurations:
```bash
python main_profile.py
```

### 2. Model Training

#### Automated Batch Training Pipeline:
To sequentially train baseline, ablation, and the main IRS-DETR models:
```bash
python train_pipeline.py
```

#### Single Model Training:
```python
from ultralytics import RTDETR

model = RTDETR('EXP-Model/IRS-DETR.yaml')
model.train(
    data='dataset/ISDD.yaml',
    imgsz=640,
    epochs=300,
    batch=4,
    workers=4,
    device=0,
    project='runs/train',
    name='IRS-DETR',
)
```

### 3. Model Evaluation
To evaluate a trained checkpoint on the test set:
```bash
python val.py
```
*(Make sure to specify your checkpoint path `model_path` in `val.py`)*

### 4. Inference & Detection
To run inference on test images:
```bash
python detect.py
```
*(Make sure to specify your checkpoint path `model_path` and image source in `detect.py`)*

```
