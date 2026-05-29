import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('your_model.pt') # select your model.pt path
    model.predict(source='dataset/images/test',
                  conf=0.25,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  show_conf=False, 
                  show_labels=False,
                  )
