import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model and face detector
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict_emotion_from_frame(frame):
    img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return "No Face Detected", frame, 'emojis/NofaceDetected.jpeg'

    (x, y, w, h) = faces[0]
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48)) / 255.0
    roi = roi.reshape(1, 48, 48, 1)

    pred = model.predict(roi)
    idx = pred.argmax()
    emotion = emotions[idx]

    emoji_path = f'emojis/{emotion}.jpg'
    return emotion, frame, emoji_path

with gr.Blocks() as demo:
    gr.Markdown("## 😄 Emojifying Facial Expressions (Snapshot via Webcam)")
    with gr.Row():
        cam_input = gr.Camera(label="Capture Your Face")
        label_output = gr.Label(label="Predicted Emotion")
        emoji_output = gr.Image(label="Emoji")
    live_image = gr.Image(label="Input Frame")

    cam_input.change(fn=predict_emotion_from_frame, inputs=cam_input, outputs=[label_output, live_image, emoji_output])

if __name__ == "__main__":
    demo.launch()
