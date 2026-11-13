# IRS-DETR: Efficient Infrared Ship Detection Transformer for Complex Sea–Air Backgrounds

## 📂 Project Structure

```text
IRS-DETR/
├── EXP-Model/                     
│   ├── IRS-DETR.yaml             
│   ├── RT-DETR.yaml               
│   └── Ablation-exp-model/        
│       ├── EFAModule.yaml         
│       └── WaveletSample.yaml     
├── dataset/                      
│   └── ISDD.yaml                 
├── ultralytics/                   
│   ├── nn/
│   │   ├── extra_modules/        
│   │   └── tasks.py               
│   └── ...
├── train_pipeline.py              
├── main_profile.py                
├── val.py                        
├── detect.py                     
├── requirements.txt               
└── setup.py                       

---

## 🛠️ Environment & Installation

### Requirements
- **OS:** Linux
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

### 4. Inference & Detection
To run inference on test images:
```bash
python detect.py
```
