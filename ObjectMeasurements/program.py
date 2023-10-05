from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import streamlit as st


model = YOLO("yolov8n.pt")

classNames = ["person","bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

prev_frame_time = 0
new_frame_time = 0
with st.sidebar:
    add_select = st.selectbox(
        "Select Activity",
        ("Home", "Webcam Object Detection")
    )
if add_select == "Home":
    st.header(":blue[Welcome To Our Project]")
    st.subheader("Object Detection and Measurement Application Using OpenCV and Streamlit")
    st.text('Welcome')
elif add_select == "Webcam Object Detection":
    st.header(":blue[Welcome To Object Detection and Measurement Application]")
    st.text("Press start to open camera and begin detecting.")
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    frame_placeholder = st.empty()
    start_button_pressed = st.button("Start")
    stop_button_pressed = st.button("Stop")
    while start_button_pressed and not stop_button_pressed:
        new_frame_time = time.time()
        success, img = cap.read()
        if not success:
            st.write("The video capture has ended")
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img_rgb, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(img_rgb, f'{classNames[cls]} W:{w / 10}cm H:{h / 10}cm', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)
        frame_placeholder.image(img_rgb, channels="RGB")
    while stop_button_pressed:
        st.text("Video Capture Ended")
        break
#streamlit run program.py
