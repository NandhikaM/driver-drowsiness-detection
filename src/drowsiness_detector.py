import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# -------------------------------
# Load trained CNN model safely
# -------------------------------
# Get the folder where this script is located
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "../models/eye_model.h5")
model = load_model(model_path)

# -------------------------------
# Load OpenCV Haar cascades
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# -------------------------------
# Parameters
# -------------------------------
IMG_SIZE = 24
closed_frames = 0
CLOSED_THRESHOLD = 3  # consecutive frames to trigger alert

# -------------------------------
# Open webcam
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray)

        eyes_status = []

        for (ex, ey, ew, eh) in eyes:
            eye_img = face_gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
            eye_img = eye_img / 255.0
            eye_img = eye_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            prediction = model.predict(eye_img, verbose=0)
            class_index = np.argmax(prediction)

            if class_index == 0:  # closed
                eyes_status.append("closed")
            else:
                eyes_status.append("open")

            # Draw rectangle around eyes
            color = (0, 0, 255) if class_index == 0 else (0, 255, 0)
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)

        # Check if all detected eyes are closed
        if len(eyes_status) > 0 and all(status == "closed" for status in eyes_status):
            closed_frames += 1
        else:
            closed_frames = 0

        # Show alert if eyes closed for enough frames
        if closed_frames >= CLOSED_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            print("DROWSINESS ALERT! Eyes closed for several frames.")

    # Show webcam feed
    cv2.imshow("Drowsiness Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()