import warnings, os, traceback, time
from pathlib import Path

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

"""IRS-DETR 训练管线"""

# ============================================================
# 配置文件
# ============================================================
DATA_YAML = 'dataset/ISDD.yaml'
IMGSZ = 640
EPOCHS = 300
BATCH = 4
WORKERS = 4
PATIENCE = 0
DEVICE = 0
BASE_PROJECT = 'runs/train'

# ============================================================
# Pipeline
# ============================================================
EXP_DIR = Path('Exp-model')


def discover_yamls():
    """Discover all YAML configs, classified by category."""
    categories = {
        'ablation': [],
        'comparison': [],
        'main': [],
    }
    for yaml_path in sorted(EXP_DIR.rglob('*.yaml')):
        rel = yaml_path.relative_to(EXP_DIR)
        parts = rel.parts
        if len(parts) > 1:
            category = parts[0]  # 'ablation-exp-model' or 'comparison-backbone'
            if category.startswith('ablation'):
                categories['ablation'].append(yaml_path)
            elif category.startswith('comparison'):
                categories['comparison'].append(yaml_path)
            else:
                categories['main'].append(yaml_path)
        else:
            categories['main'].append(yaml_path)
    return categories


def train_one(config_path, project, name):
    """Train a single model config."""
    print(f"\n{'#'*70}")
    print(f"# Training: {name}")
    print(f"# Config:  {config_path}")
    print(f"# Project: {project}")
    print(f"{'#'*70}\n")

    model = RTDETR(str(config_path))
    model.train(
        data=DATA_YAML,
        cache=False,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        patience=PATIENCE,
        project=project,
        name=name,
    )


def main():
    categories = discover_yamls()

    total = sum(len(v) for v in categories.values())
    current = 0
    results = {'passed': [], 'failed': []}

    # 1. Main configs (IRS-DETR, RT-DETR-R18)
    for yaml_path in categories['main']:
        current += 1
        name = yaml_path.stem
        project = f'{BASE_PROJECT}/main'
        start = time.time()
        try:
            train_one(yaml_path, project, name)
            elapsed = time.time() - start
            print(f"\n[DONE {current}/{total}] {name} ({elapsed:.0f}s)")
            results['passed'].append(str(yaml_path))
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n[FAIL {current}/{total}] {name}: {e} ({elapsed:.0f}s)")
            traceback.print_exc()
            results['failed'].append(str(yaml_path))

    # 2. Ablation experiments
    for yaml_path in categories['ablation']:
        current += 1
        name = yaml_path.stem
        project = f'{BASE_PROJECT}/ablation'
        start = time.time()
        try:
            train_one(yaml_path, project, name)
            elapsed = time.time() - start
            print(f"\n[DONE {current}/{total}] {name} ({elapsed:.0f}s)")
            results['passed'].append(str(yaml_path))
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n[FAIL {current}/{total}] {name}: {e} ({elapsed:.0f}s)")
            traceback.print_exc()
            results['failed'].append(str(yaml_path))

    # 3. Comparison backbone experiments
    for yaml_path in categories['comparison']:
        current += 1
        name = yaml_path.stem
        project = f'{BASE_PROJECT}/comparison'
        start = time.time()
        try:
            train_one(yaml_path, project, name)
            elapsed = time.time() - start
            print(f"\n[DONE {current}/{total}] {name} ({elapsed:.0f}s)")
            results['passed'].append(str(yaml_path))
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n[FAIL {current}/{total}] {name}: {e} ({elapsed:.0f}s)")
            traceback.print_exc()
            results['failed'].append(str(yaml_path))

    # ============================================================
    # 训练结果总结
    # ============================================================
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Passed: {len(results['passed'])}/{total}")
    for p in results['passed']:
        print(f"  [OK]  {p}")
    if results['failed']:
        print(f"Failed: {len(results['failed'])}/{total}")
        for p in results['failed']:
            print(f"  [FAIL] {p}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
