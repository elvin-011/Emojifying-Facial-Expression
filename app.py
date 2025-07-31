import gradio as gr
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load pre-trained model and cascade
model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "No Face", 'emojis/NofaceDetected.jpeg'
    
    (x, y, w, h) = faces[0]
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48)) / 255.0
    roi = roi.reshape(1, 48, 48, 1)

    pred = model.predict(roi)
    idx = pred.argmax()
    emotion = emotions[idx]

    emoji_path = f'emojis/{emotion}.jpg'
    return emotion, emoji_path

with gr.Blocks() as demo:
    gr.Markdown("# 😊 Real-time Emoji Detection")
    cam = gr.Image(source="webcam", streaming=True, label="Live Feed")
    label = gr.Label(label="Predicted Emotion")
    emoji = gr.Image(label="Emoji Output")

    cam.stream(fn=detect_emotion, inputs=cam, outputs=[label, emoji])

demo.launch()
