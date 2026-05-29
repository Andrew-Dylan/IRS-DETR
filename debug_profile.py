import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR

# 所有待测配置文件
config_files = [
    r'Exp-model\comparison-backbone\rtdetr-EfficientViT.yaml',
    r'Exp-model\comparison-backbone\rtdetr-fasternet.yaml',
    r'Exp-model\comparison-backbone\rtdetr-mobilenetv4.yaml',
    r'Exp-model\comparison-backbone\rtdetr-r18.yaml',
    r'Exp-model\comparison-backbone\rtdetr-repvit.yaml',
    r'Exp-model\comparison-backbone\rtdetr-RMT.yaml',
    r'Exp-model\comparison-backbone\rtdetr-SwinTiny.yaml',
]

if __name__ == '__main__':
    for config_path in config_files:
        config_name = config_path.split('\\')[-1]
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print('='*60)
        try:
            model = RTDETR(config_path)
            print("Model created successfully")
            model.model.eval()
            print("Model set to eval mode")
            model.info(detailed=True)
            print("Model info printed")
            try:
                print("Running profile...")
                model.profile(imgsz=[640, 640])
                print("Profile completed")
            except Exception as e:
                import traceback
                print(f"Profile error: {e}")
                traceback.print_exc()
            model.fuse()
            print(f"[PASS] {config_name}")
        except Exception as e:
            import traceback
            print(f"[FAIL] {config_name}: {e}")
            traceback.print_exc()
        print()