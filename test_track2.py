import cv2
from ultralytics import YOLO

model = YOLO('yolov8x-pose.pt')
results = model.track(source="https://www.youtube.com/watch?v=CFOQXmhCnK0", stream=True, show=True)
