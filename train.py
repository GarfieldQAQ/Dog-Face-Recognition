# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'/mnt/d/Code/yolo/yolov12/yolov12s.pt')
    model.load('/mnt/c/Users/Cyber/Desktop/work3/runs/train/exp5/weights/best.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=20,
                batch=4,
                workers=6,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='./runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
