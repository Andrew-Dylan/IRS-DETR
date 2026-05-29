import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR
import os
from pathlib import Path

# 所有待测配置文件（自动发现，避免漏配或路径拼写错误）
EXP_DIR = Path('Exp-model')
config_files = [str(p) for p in sorted(EXP_DIR.rglob('*.yaml'))]

if __name__ == '__main__':
    for config_path in config_files:
        config_name = os.path.basename(config_path)
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print('='*60)
        try:
            model = RTDETR(config_path)
            model.model.eval()
            model.info(detailed=True)
            try:
                model.profile(imgsz=[640, 640])
            except Exception as e:
                print(f"Profile error: {e}")
            model.fuse()
            print(f"[PASS] {config_name}")
        except Exception as e:
            print(f"[FAIL] {config_name}: {e}")
        print()
        