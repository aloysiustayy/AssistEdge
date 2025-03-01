import threading
from flask import Flask, jsonify
import paho.mqtt.client as mqtt
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)
CORS(app)  


# Global storage for messages received from two Raspberry Pis
data_store = {
    "sign_language": [],  # Data from sign language Raspberry Pi
    "emotion": []         # Data from emotion Raspberry Pi
}

# MQTT Broker configuration
MQTT_BROKER = "raspberrypi.local"  # Replace with your MQTT broker's IP address
MQTT_PORT = 1883
TOPIC_SIGN = "assistedge/sign_language"
TOPIC_EMOTION = "assistedge/emotion"

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code", rc)
    # Subscribe to both topics
    client.subscribe(TOPIC_SIGN)
    client.subscribe(TOPIC_EMOTION)

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode()
    print(f"Received message on {topic}: {payload}")
    # Store message based on topic
    if topic == TOPIC_SIGN:
        data_store["sign_language"].append(payload)
        # Limit stored messages to last 100
        if len(data_store["sign_language"]) > 100:
            data_store["sign_language"] = data_store["sign_language"][-100:]
    elif topic == TOPIC_EMOTION:
        data_store["emotion"].append(payload)
        if len(data_store["emotion"]) > 100:
            data_store["emotion"] = data_store["emotion"][-100:]

def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    # This call blocks forever, processing network events.
    client.loop_forever()

# Flask routes
@app.route('/')
def index():
    return "MQTT Flask Server Running. Visit /data to see the latest messages."

@app.route('/data')
def get_data():
    # Return the stored MQTT data as JSON
    return jsonify(data_store)

@app.route('/sign-language')
def get_translated_sign():
    return jsonify({'sign_language': ['hello world! :0', 'hi', 'no']})
    # return jsonify(data_store['sign_language'])


if __name__ == '__main__':
    # Start the MQTT client in a separate thread
    thread = threading.Thread(target=mqtt_thread, daemon=True)
    thread.start()
    
    # Run Flask app on all interfaces on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
