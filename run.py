import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and face detector
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def plot_image(img, emoj):
    hmin = 200
    wmin = 200
    img = cv2.resize(img, (wmin, hmin))
    emoj = cv2.resize(emoj, (wmin, hmin))
    combined = np.hstack((img, emoj))
    cv2.imshow('Emotion Recognition', combined)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(1)
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48)) / 255.0
        roi = roi.reshape(1, 48, 48, 1)
        pred = model.predict(roi)
        idx = pred.argmax()
        emotion = emotions[idx]

        emoj = cv2.imread(f'emojis/{emotion}.jpg')
        plot_image(frame, emoj)

    else:
        emoj = cv2.imread('NofaceDetected.jpeg')
        plot_image(frame, emoj)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
