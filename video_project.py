import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import webbrowser

class ObjectDetector(VideoTransformerBase):
    def __init__(self):
        self.thres = 0.4  # Threshold to detect object
        self.classNames = []
        self.classFile = 'coco.names'
        with open(self.classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        self.configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.weightsPath = 'frozen_inference_graph.pb'

        self.net = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.group_invite_opened = False  # Flag to track if the group invite link has been opened

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        classIds, confs, bbox = self.net.detect(img, confThreshold=self.thres)

        if not self.group_invite_opened:  # Check if invite link hasn't been opened
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if 44 in classIds:
                        webbrowser.open_new_tab("https://chat.whatsapp.com/BGtWD7jDZbXEo5BMqAjpOn")
                        self.group_invite_opened = True  # Set flag to True after opening link
                        break

                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=6)
                    cv2.putText(img, self.classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return img

st.title('Machine Learning Club - Object Detection')
st.write('Show a bottle to join our WhatsApp group!') 

webrtc_ctx = webrtc_streamer(key="object-detection", video_transformer_factory=ObjectDetector)

if webrtc_ctx.video_transformer:
    st.write('Object detection is active')

# Thank you message after the invite link has been opened
if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.group_invite_opened:
    st.write('Thank you for joining our WhatsApp group!')
    st.stop()  # Stop the app to prevent further processing
