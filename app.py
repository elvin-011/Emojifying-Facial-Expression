import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and cascade
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        return image  # fallback if emoji image not found

    # Resize and combine
    emoji_img = cv2.resize(emoji_img, (image.shape[1], image.shape[0]))
    combined = np.hstack((image, emoji_img))
    return combined

iface = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(type="numpy", label="Upload a Face Image"),
    outputs=gr.Image(type="numpy", label="Image + Emoji"),
    title="Emojifying Facial Expression",
    description="Upload an image of a face to see the corresponding emoji based on detected emotion."
)

if __name__ == "__main__":
    iface.launch()
