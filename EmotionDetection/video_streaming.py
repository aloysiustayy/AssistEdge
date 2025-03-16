import cv2
import base64
import socketio
import argparse
import time
from deepface import DeepFace



parser = argparse.ArgumentParser(description="Run emotion detection with optional GUI")
parser.add_argument("--headless", action="store_true", help="Disable GUI display")
parser.add_argument("--check", type=int, default=1, help="Run DeepFace analysis every N frames (default: every frame)")
parser.add_argument("--fps", type=int, default=10, help="Set the target FPS for stable performance (default: 10)")
parser.add_argument("--ip", type=str, default="172.20.10.3", help="Flask Server IP address")
args = parser.parse_args()

# Connect to the Flask-SocketIO server
sio = socketio.Client()
FLASK_SERVER_URL = f"http://{args.ip}:5001" #"http://192.168.18.20:5001"  # Replace with your server's IP if needed

@sio.event
def connect():
    print("Successfully connected to Flask-SocketIO server.")

@sio.event
def connect_error(data):
    print("Failed to connect to Flask-SocketIO server.")

@sio.event
def disconnect():
    print("Disconnected from Flask-SocketIO server.")

sio.connect(FLASK_SERVER_URL, namespaces=["/"])
print(f"Connecting to FLASK_SERVER_URL at {FLASK_SERVER_URL}")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Set a lower resolution for smoother processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_counter = 0
emotion = "unknown"

target_frame_time = 1.0 / args.fps  # Convert FPS to frame time in seconds
last_frame_time = time.monotonic()  # Initialize the time tracker

while True:
    all_emotions = {}
    current_time = time.monotonic()
    elapsed_time = current_time - last_frame_time

    # Skip frame if we're processing too fast
    if elapsed_time < target_frame_time:
        continue  # Skip to next iteration without blocking

    last_frame_time = current_time  # Update time tracker

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to grayscale and then to RGB
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Run DeepFace analysis every N frames as per the check argument
    if frame_counter % args.check == 0:
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if result and isinstance(result, list):
                    emotion = result[0].get('dominant_emotion', 'unknown')
                    if emotion not in all_emotions:
                        all_emotions[emotion] = 1
                    else:
                        all_emotions[emotion] += 1
            except Exception as e:
                print(f"Error analyzing emotion: {e}")
                emotion = "error"

    # Draw bounding boxes and emotion text on all detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Encode the annotated frame to Base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    # Emit the frame and emotion via WebSocket
    # sio.emit('frame', {"frame": frame_base64, "emotion": emotion})
    
    if sio.connected:
        sio.emit('frame', {"frame": frame_base64, "all_emotion": all_emotions}, namespace="/")
    else:
        print("SocketIO not connected, skipping emit.")
    # Optionally display the frame locally if not in headless mode
    if not args.headless:
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
sio.disconnect()

