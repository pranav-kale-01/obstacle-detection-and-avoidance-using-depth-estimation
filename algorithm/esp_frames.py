import requests
import cv2

def get_video(input_id):
    camera = cv2.VideoCapture(input_id)
    requests.get("http://192.168.235.55/control?var=framesize&val=11")
    requests.get("http://192.168.235.55/control?var=led_intensity&val=20")
    
    while True:
        okay, frame = camera.read()
        if not okay:
            break

        cv2.imshow('video', frame)
        cv2.waitKey(1)
    pass

if __name__ == '__main__':
    get_video("http://192.168.235.55:81/stream")