from flask import Flask, request
import cv2
import numpy as np
import cv2
import os
from yolov8 import YOLOv8

app = Flask(__name__)
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.5)
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        boxes, scores, class_ids = yolov8_detector(img)

        # Draw detections
        combined_img = yolov8_detector.draw_detections(img)
        os.makedirs("./doc/img", exist_ok=True)
        # cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        # cv2.imshow("Detected Objects", combined_img)
        cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
        cv2.waitKey(0)
    return "Frame Received", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
