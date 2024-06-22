import cv2
import requests
import threading

def capture_video(cap, server_url):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Encode frame to JPEG
        _, img_encoded = cv2.imencode('.jpg', frame)
        # Send frame to the server in a separate thread
        threading.Thread(target=send_frame, args=(server_url, img_encoded)).start()
        
        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def send_frame(server_url, img_encoded):
    try:
        response = requests.post(server_url, data=img_encoded.tobytes())
        print(response.status_code, response.reason)
    except Exception as e:
        print("Exception while sending frame:", e)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    ec2_ip = 'YOUR_EC2_PUBLIC_IP'
    server_url = f'http://{ec2_ip}:5000/upload_frame'
    
    capture_video(cap, server_url)
    
    cap.release()
    cv2.destroyAllWindows()
