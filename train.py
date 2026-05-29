import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('your_model.yaml') # 模型配置文件路径
    model.train(data='dataset/ISDD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                patience=0,
                project='runs/train',
                name='exp',
                )