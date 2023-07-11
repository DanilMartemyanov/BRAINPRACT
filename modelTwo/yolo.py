

import torch

# Load the YOLOv5 model

model = torch.hub.load("ultralytics/yolov5", 'custom', path='weight/yolov5.pt')

def yolo_for_image(file):
    return model(file)