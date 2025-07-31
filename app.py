import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# Load the model (ensure it's small enough for Hugging Face Spaces)
model = load_model("model.h5")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the emoji images in memory
emoji_images = {emotion: cv2.imread(f"emojis/{emotion}.jpg") for emotion in emotions}
no_face_img = cv2.imread("NofaceDetected.jpeg")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = roi.reshape(1, 48, 48, 1)
        pred = model.predict(roi, verbose=0)
        idx = pred.argmax()
        emoji = emoji_images[emotions[idx]]
    else:
        emoji = no_face_img

    frame_resized = cv2.resize(frame, (200, 200))
    emoji_resized = cv2.resize(emoji, (200, 200))
    combined = np.hstack((frame_resized, emoji_resized))
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=gr.Image(label="Live Emotion Recognition"),
    live=True,
    title="Real-Time Emoji Emotion Detector"
).launch()
