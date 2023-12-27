import os
import threading
import time

import av
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import (Conv2D)
from keras.layers import (MaxPooling2D)
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# TODO clean up code

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Angry",
    1: "Disguested",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dist = {
    0: cur_path+"/data/emojis/angry.png",
    1: cur_path+"/data/emojis/disgusted.png",
    2: cur_path+"/data/emojis/fearful.png",
    3: cur_path+"/data/emojis/happy.png",
    4: cur_path+"/data/emojis/neutral.png",
    5: cur_path+"/data/emojis/sad.png",
    6: cur_path+"/data/emojis/surprised.png",
}

show_text=[0]
def show_subject(image):
    frame1 = cv2.resize(image, (600, 500))
    bounding_box = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1,
                    emotion_dict[maxindex],
                    (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4)
        show_text[0]=maxindex

def show_avatar():
    text = emotion_dict[show_text[0]]
    frame2= cv2.imread(emoji_dist[show_text[0]])
    return frame2, text

st.title("Webcam Live Feed")
class VideoProcessor():
    frame_lock: threading.Lock
    out_image: np.ndarray

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.out_image = None
        self.last_capture = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with self.frame_lock:
            self.out_image = img

            # Capture an image every second.
            if time.time() - self.last_capture > 1:
                cv2.imwrite(f'frame_{int(self.last_capture)}.jpg', img)
                self.last_capture = time.time()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    ctx = webrtc_streamer(
        key="snapshot",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

    image_placeholder = st.empty()
    text_placeholder = st.empty()

    while True:
        if ctx.video_processor:
            with ctx.video_processor.frame_lock:
                out_image = ctx.video_processor.out_image
            if out_image is not None:
                show_subject(out_image)
                avatar, text = show_avatar()
                text_placeholder.text(text)
                image_placeholder.image(avatar, channels="BGR", caption='Avatar')
            else:
                st.warning("No frames available yet.")
        time.sleep(2)

if __name__ == "__main__":
    main()