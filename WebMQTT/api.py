import threading
import asyncio
from hbmqtt.broker import Broker
import paho.mqtt.client as mqtt
from flask import Flask, jsonify
from flask_cors import CORS

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
    }
}

broker = Broker(broker_config)

async def start_broker():
    await broker.start()

def broker_thread():
    # Run the broker using asyncio in this thread
    asyncio.run(start_broker())

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
# Use localhost since the broker runs in this combined script.
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
    client.loop_forever()

# ---------------------------
# Flask Webserver Setup
# ---------------------------
app = Flask(__name__)
CORS(app)

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
# Main: Start Broker, MQTT Client, and Flask App
# ---------------------------
if __name__ == '__main__':
    # Start the MQTT broker in its own thread.
    broker_thread_obj = threading.Thread(target=broker_thread, daemon=True)
    broker_thread_obj.start()

    # Start the MQTT client in another thread.
    client_thread = threading.Thread(target=mqtt_client_thread, daemon=True)
    client_thread.start()

    # Run the Flask webserver (listening on port 5001).
    app.run(host='0.0.0.0', port=5001, debug=True)
