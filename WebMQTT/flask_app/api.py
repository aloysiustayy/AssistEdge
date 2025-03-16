import threading
import multiprocessing
import asyncio
import base64
import cv2
import numpy as np
from hbmqtt.broker import Broker
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from flask_socketio import SocketIO
import csv
import time
import psutil
from datetime import datetime
import json
from flask import Flask, Response
import os
# ---------------------------
# Global Data Storage
# ---------------------------
combined_data_store = {}  # Stores data from both MQTT and SocketIO before saving
csv_filename = "combined_data.csv"
last_appended_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def is_within_time_difference(timestamp1, timestamp2, x):
    """
    Check if the difference between two timestamps is less than x seconds.

    :param timestamp1: First timestamp (string in '%Y-%m-%d %H:%M:%S' format)
    :param timestamp2: Second timestamp (string in '%Y-%m-%d %H:%M:%S' format)
    :param x: Time difference threshold in seconds (integer)
    :return: True if the difference is less than x seconds, False otherwise
    """
    time1 = datetime.strptime(timestamp1, '%Y-%m-%d %H:%M:%S')
    time2 = datetime.strptime(timestamp2, '%Y-%m-%d %H:%M:%S')
    # print(f"Time diff between {time2} and {time1} is {abs((time2 - time1).total_seconds())}")
    return abs((time2 - time1).total_seconds()) < x
    
# ---------------------------
# Save Merged Data to CSV
# ---------------------------

def save_to_csv():
    """ Check for complete data pairs and save them to CSV. """
    global combined_data_store

    with open(csv_filename, "a", newline="") as file:
        writer = csv.writer(file)

        # Write header only if file is empty
        if file.tell() == 0:
            writer.writerow([
                "cur_time", "recognised_word", "recognised_emotion", 
                "recognised_speech", "cpu_usage", 
                "correlation_with_emotion", "frequent_pairs"
            ])

        for timestamp, data in list(combined_data_store.items()):
            if "recognised_word" in data and "recognised_emotion" in data:
                writer.writerow([
                    timestamp, 
                    data.get("recognised_word", "N/A"), 
                    data.get("recognised_emotion", "N/A"),
                    data.get("recognised_speech", "N/A"),
                    get_cpu_usage(),
                    data.get("correlation_with_emotion", "N/A"),
                    data.get("frequent_pairs", "N/A")
                ])

def update_csv(emotion_ts, all_emotions):
    #   flow
    #   1. detects sign language
    #   2. for the next 3 seconds, add any emotions detected to the same row

    global last_appended_timestamp
    if last_appended_timestamp in combined_data_store:
        if "recognised_emotion" not in combined_data_store[last_appended_timestamp]:
            if is_within_time_difference(last_appended_timestamp, emotion_ts, 3) and last_appended_timestamp != "":
                # print("is within time diff!!")

                combined_data_store[last_appended_timestamp].update({
                    "recognised_emotion": all_emotions
                })
                print(combined_data_store)

                save_to_csv()
                print(f"Saving {combined_data_store[last_appended_timestamp]}!")


# ---------------------------
# Function to Get CPU Usage
# ---------------------------
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

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
}

# ---------------------------
# MQTT Client (paho-mqtt) Setup
# ---------------------------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_SIGN = "assistedge/sign_language"

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code", rc)
    client.subscribe(TOPIC_SIGN)

def on_message(client, userdata, msg):
    # topic = msg.topic
    # payload = msg.payload.decode()
    # print(f"Received message on {topic}: {payload}")
    # if topic == TOPIC_SIGN:
        # data_store["sign_language"].append(payload)
        # if len(data_store["sign_language"]) > 100:
        #     data_store["sign_language"] = data_store["sign_language"][-100:]
   
    # new
    global last_appended_timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payload = msg.payload.decode()

    print(f"[MQTT] Received Word: {payload}")

    # Store data in shared dictionary
    if timestamp not in combined_data_store:
        combined_data_store[timestamp] = {}
    combined_data_store[timestamp].update({
        "recognised_word": payload
    })

    # assign current timestamp to global timestamp for easy reference in other processes and methods
    last_appended_timestamp = timestamp

    # working version of displaying data in react
    data_store["sign_language"].append(payload)
    if len(data_store["sign_language"]) > 100:
        data_store["sign_language"] = data_store["sign_language"][-100:]


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
# Flask & SocketIO Webserver Setup
# ---------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables to hold the latest frame and emotion counts
latest_frame = None
latest_emotion_counts = {}

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """
    Receives video frames & emotion counts from Raspberry Pi,
    updates the global variables, and broadcasts them via SocketIO.
    """
    global latest_frame, latest_emotion_counts
    try:
        data = request.json
        frame_base64 = data.get("frame", "")
        latest_emotion_counts = data.get("emotion_counts", {})

        if frame_base64:
            # Decode and update the latest frame
            img_bytes = base64.b64decode(frame_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            latest_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # Broadcast the frame and emotion counts to all SocketIO clients
            socketio.emit('new_frame', {'frame': frame_base64, 'emotion_counts': latest_emotion_counts})
        return "Frame received", 200
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/video_feed')
def video_feed():
    """
    Serves a video stream by yielding JPEG-encoded frames.
    """
    def generate():
        while True:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_counts')
def get_emotion_counts():
    """Returns the latest emotion counts as JSON."""
    return jsonify(latest_emotion_counts), 200

@app.route('/data')
def get_data():
    """Returns the latest MQTT messages data."""
    return jsonify(data_store)

@app.route('/sign-language')
def get_translated_sign():
    """Returns the latest sign language messages."""
    return jsonify({'sign_language': data_store["sign_language"]})

@app.route('/clear', methods=['POST'])
def clear_sign_language():
    data_store["sign_language"].clear()
    return jsonify({"status": "success", "message": "Sign language data cleared"})


@app.route('/dashboard')
def get_dashboard():
    global combined_data_store
    return jsonify(combined_data_store)

# Serve the CSV file
@app.route('/csv')
def get_csv():
    csv_path = "combined_data.csv"
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        return f"File not found: {csv_path}", 404
    
    # Read the CSV file into memory
    with open(csv_path, 'r') as file:
        csv_data = file.read()
    
    # Return the CSV data as a response
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=combined_data.csv"}
    )

@socketio.on('connect')
def handle_connect():
    print('SocketIO client connected.')

@socketio.on('disconnect')
def handle_disconnect():
    print('SocketIO client disconnected.')

@socketio.on('frame')
def handle_frame(data):
    # Optionally re-emit the frame data if needed
    socketio.emit('new_frame', data)

    """Handles data from Raspberry Pi (Emotion Recognition & Speech-to-Text)."""
    global combined_data_store

    recognised_emotion = data.get("all_emotion", "N/A")
    if recognised_emotion != {}:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_csv(timestamp,recognised_emotion)

# ---------------------------
# Main: Start Broker, MQTT Client, and Flask/SocketIO Server
# ---------------------------
if __name__ == '__main__':
    # Start the MQTT broker in a separate process.
    broker_process = multiprocessing.Process(target=run_broker, daemon=True)
    broker_process.start()
    print("MQTT Broker process started.")

    # Start the MQTT client in a separate thread.
    client_thread = threading.Thread(target=mqtt_client_thread, daemon=True)
    client_thread.start()

    # Run the Flask app with SocketIO.
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)

    # If the Flask app terminates, clean up the broker process.
    broker_process.terminate()

