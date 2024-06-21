import cv2

from yolov8 import YOLOv8

# # Initialize video
# cap = cv2.VideoCapture("input.mp4")

videoUrl = 'sample_data/hardhat.mp4'

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

# Initialize YOLOv7 model
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.5)
cap = cv2.VideoCapture(videoUrl)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)
    # out.write(combined_img)

# out.release()