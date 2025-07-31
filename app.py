import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and cascade
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(image):
    if image is None:
        return None

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48)) / 255.0
        roi = roi.reshape(1, 48, 48, 1)
        pred = model.predict(roi)
        idx = pred.argmax()
        emotion = emotions[idx]
        emoji_img = cv2.imread(f'emojis/{emotion}.jpg')
    else:
        emoji_img = cv2.imread('NofaceDetected.jpeg')
    
    if emoji_img is None:
        return image_bgr

    emoji_img = cv2.resize(emoji_img, (image_bgr.shape[1], image_bgr.shape[0]))
    combined = np.hstack((image_bgr, emoji_img))
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    return combined_rgb

iface = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(source="webcam", label="Capture Your Face (Click 'Capture')"),
    outputs=gr.Image(type="numpy", label="Image + Emoji"),
    title="Emojifying Facial Expression",
    description="Capture a face using webcam or upload an image to see the corresponding emoji based on emotion detected."
)

if __name__ == "__main__":
    iface.launch()
