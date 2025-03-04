import cv2
import json
import base64
import requests
from deepface import DeepFace

FLASK_SERVER_URL = "http://192.168.1.111:5001/upload_frame"  # Flask endpoint

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    emotion = "unknown"

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            # Perform emotion analysis
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if result and isinstance(result, list):
                emotion = result[0].get('dominant_emotion', 'unknown')
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            emotion = "error"

        # Draw bounding box & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Convert frame to Base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    # Send frame to Flask server
    try:
        requests.post(FLASK_SERVER_URL, json={"frame": frame_base64, "emotion": emotion})
    except Exception as e:
        print(f"Error sending frame: {e}")

    # Display frame locally
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
