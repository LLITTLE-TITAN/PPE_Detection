from flask import Flask, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imshow('Received Frame', img)
        cv2.waitKey(1)
    return "Frame Received", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
