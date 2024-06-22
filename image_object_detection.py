import cv2
import os
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.5)
img_path = "sample_data/two-young-construction-workers-wearing-555864.jpg"
# Read image
img = cv2.imread(img_path)

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
os.makedirs("./doc/img",exist_ok=True)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
