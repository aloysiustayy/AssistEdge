import threading
import multiprocessing
import asyncio
from hbmqtt.broker import Broker
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import base64
import cv2
import numpy as np

# ---------------------------
# MQTT Broker (HBMQTT) Setup
# ---------------------------
broker_config = {
    'listeners': {
        'default': {
            'type': 'tcp',
            'bind': '0.0.0.0:1883'  # Listen on all interfaces, port 1883
        }
    },
    'sys_interval': 10,
    'auth': {
        'allow-anonymous': True
    },
    'topic-check': {   # Disable topic-checking to avoid warnings
        'enabled': False
    }
}

def run_broker():
    # This function runs in a separate process.
    from hbmqtt.broker import Broker  # Re-import in the child process
    broker = Broker(broker_config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(broker.start())
        loop.run_forever()
    except Exception as e:
        print("Broker startup failed:", e)
    finally:
        loop.close()

# ---------------------------
# Global Data Store for MQTT Messages
# ---------------------------
data_store = {
    "sign_language": [],  # Data from sign language Raspberry Pi
    "emotion": []         # Data from emotion Raspberry Pi
}

# ---------------------------
# MQTT Client (paho-mqtt) Setup
# ---------------------------
# Use localhost since the broker runs in the same machine (in a separate process).
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_SIGN = "assistedge/sign_language"
TOPIC_EMOTION = "assistedge/emotion"

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code", rc)
    client.subscribe(TOPIC_SIGN)
    client.subscribe(TOPIC_EMOTION)

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode()
    print(f"Received message on {topic}: {payload}")
    if topic == TOPIC_SIGN:
        data_store["sign_language"].append(payload)
        if len(data_store["sign_language"]) > 100:
            data_store["sign_language"] = data_store["sign_language"][-100:]
    elif topic == TOPIC_EMOTION:
        data_store["emotion"].append(payload)
        if len(data_store["emotion"]) > 100:
            data_store["emotion"] = data_store["emotion"][-100:]

def mqtt_client_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print("Error connecting to MQTT broker:", e)
        return
    client.loop_forever()

# ---------------------------
# Flask Webserver Setup
# ---------------------------
app = Flask(__name__)
CORS(app)

latest_frame = None
latest_emotion_counts = {}  # Stores count of each emotion

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Receives video frames & emotion counts from Raspberry Pi."""
    global latest_frame, latest_emotion_counts
    try:
        data = request.json
        frame_base64 = data.get("frame", "")
        latest_emotion_counts = data.get("emotion_counts", {})

        if frame_base64:
            img_bytes = base64.b64decode(frame_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            latest_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return "Frame received", 200
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/video_feed')
def video_feed():
    """Serves video stream to React."""
    def generate():
        while True:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_counts')
def get_emotion_counts():
    """Sends the latest emotion count data to React."""
    return latest_emotion_counts, 200

@app.route('/')
def index():
    return "MQTT Flask Server Running. Visit /data to see the latest messages."

@app.route('/data')
def get_data():
    return jsonify(data_store)

@app.route('/sign-language')
def get_translated_sign():
    return jsonify({'sign_language': data_store["sign_language"]})

# ---------------------------
# Main: Start Broker (in separate process), MQTT Client (in thread), and Flask App
# ---------------------------
if __name__ == '__main__':
    # Start the MQTT broker in a separate process.
    # broker_process = multiprocessing.Process(target=run_broker, daemon=True)
    # broker_process.start()
    # print("Broker process started.")

    # Start the MQTT client in a separate thread.
    client_thread = threading.Thread(target=mqtt_client_thread, daemon=True)
    client_thread.start()

    # Run the Flask webserver (listening on port 5001).
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

    # When the Flask app terminates, you might want to terminate the broker process.
    broker_process.terminate()
