import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import mediapipe as mp
import paho.mqtt.client as mqtt
import time
import netifaces
import ipaddress
import nmap
import threading
import queue
import pyttsx3
import argparse
import sys
import subprocess
import platform

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description='Real-time Sign Language Inference with TTS and MQTT.')
parser.add_argument('--headless', action='store_true', help='Run without OpenCV GUI windows')
args = parser.parse_args()
headless = args.headless

# =============================
# Network Discovery Functions (Optional)
# =============================
def get_network_range(interface='en0'):
    try:
        addrs = netifaces.ifaddresses(interface)
        inet_info = addrs[netifaces.AF_INET][0]
        ip_addr = inet_info['addr']
        netmask = inet_info['netmask']
        network = ipaddress.IPv4Network(f"{ip_addr}/{netmask}", strict=False)
        return str(network)
    except Exception as e:
        print("Error retrieving network range:", e)
        return None

def scan_network(network_range):
    nm = nmap.PortScanner()
    print(f"Scanning network range: {network_range}...")
    nm.scan(hosts=network_range, arguments="-sn")
    
    devices = []
    for host in nm.all_hosts():
        hostname = "Unknown"
        if 'hostnames' in nm[host] and nm[host]['hostnames']:
            hostname = nm[host]['hostnames'][0]['name'] or "Unknown"
        devices.append((hostname, host))
    return devices

def choose_broker_ip():
    network_range = get_network_range('en0')
    if not network_range:
        print("Could not determine network range. Ensure 'en0' is active.")
        return None
    
    devices = scan_network(network_range)
    if not devices:
        print("No devices found on the network.")
        return None
    
    print("\nDiscovered devices:")
    for idx, (hostname, ip) in enumerate(devices, start=1):
        print(f"{idx}. Hostname: {hostname}, IP: {ip}")
    
    try:
        choice = int(input("Enter the number of the device to use as MQTT_BROKER: "))
        if 1 <= choice <= len(devices):
            selected_ip = devices[choice - 1][1]
            print(f"Selected MQTT_BROKER IP: {selected_ip}")
            return selected_ip
        else:
            print("Invalid selection.")
            return None
    except Exception as e:
        print("Error processing selection:", e)
        return None

# =============================
# TFLite Model Setup
# =============================
MODEL_PATH = "sign_mnist_model_20epoch.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # Expected: [1, 28, 28, 1]
input_dtype = input_details[0]['dtype']

print(f"Input DType: {input_dtype}")
print("Quantization parameters:", input_details[0]['quantization'])

# Define class names (adjust if your mapping is different)
class_names = ["A", "B", "C", "D", "E",
               "F", "G", "H", "I", "J",
               "K", "L", "M", "N", "O",
               "P", "Q", "R", "S", "T",
               "U", "V", "W", "X", "Y"]

# =============================
# MediaPipe Hands Setup
# =============================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=1,
                                min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# =============================
# Preprocessing Function
# =============================
def preprocess(image):
    """
    Preprocess the input image to match the TFLite model's input:
    - Convert to grayscale.
    - Resize to 28x28.
    - Expand dimensions to shape (1,28,28,1).
    - Normalize to [0,1] if the model expects float input.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    resized = np.expand_dims(resized, axis=-1)
    input_data = np.expand_dims(resized, axis=0)
    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(input_dtype)
    return input_data

# =============================
# TTS Worker Thread
# =============================
tts_queue = queue.Queue()

def speak_text(text):
    # Check if we're on an ARM architecture (common on Raspberry Pi)
    if "arm" in platform.machine().lower():
        # Use espeak-ng via subprocess on Raspberry Pi
        subprocess.run(["espeak-ng", text.lower()])
    else:
        # Use pyttsx3 on desktop/laptop (x86 systems)
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


def onEnd(name, completed):
    if name == 'word':
        engine.endLoop()
    
def tts_worker():
    engine = pyttsx3.init()
    engine.connect('finished-utterance', onEnd)
    last_spoken_time = 0
    cooldown = 2.0  # seconds between speeches
    last_text = ""
    while True:

        try:
            text = tts_queue.get()
            current_time = time.time()
            if (current_time - last_spoken_time) > cooldown and text != last_text:
                try:
                    print("Engine saying: " + text)
                    print("Platform: "+ platform.machine().lower())
                    
                    if "aarch64" in platform.machine().lower():
                        # Use espeak-ng via subprocess on Raspberry Pi
                        subprocess.run(["espeak-ng", text.lower()])
                    else:
                        engine.say(str(text).lower(), 'word')
                        engine.startLoop()
                        time.sleep(2)
                        if headless and engine._inLoop:
                            engine.endLoop()
                            print("Loop ended!!")
                except RuntimeError as e:
                    if "run loop already started" in str(e):
                        print("TTS engine run loop already started; skipping TTS for:", text)
                    else:
                        raise
                last_spoken_time = current_time
                last_text = text
        except queue.Empty:
            continue
    if headless and engine._inLoop:
        engine.endLoop()
        print("Loop ended!!")

# =============================
# Main Function: Inference, MQTT, and TTS
# =============================
def main():
    # Start the TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # Choose MQTT broker IP (or hardcode it, as shown below)
    # Uncomment the following line to scan your network:
    chosen_ip = choose_broker_ip()
    # chosen_ip = "172.20.10.8"  # Replace with your MQTT broker's IP if needed
    if not chosen_ip:
        print("No valid MQTT broker selected. Exiting.")
        return

    MQTT_BROKER = chosen_ip
    MQTT_PORT = 1883
    MQTT_TOPIC = "assistedge/sign_language"

    # Setup MQTT client
    # Setup MQTT client
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print("Error connecting to MQTT broker:", e)
        return
    mqtt_client.loop_start()

    # Initialize webcam

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    prev_predicted_index = -1
    prev_sent_mqtt = "-"
    streak_count = 0
    min_streak = 10  # Number of consecutive frames required before confirming detection

    # Define a penalty for specific classes (adjust as needed)
    penalty_classes = {"Y": 0.2, "W": 0.2}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        roi = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            img_h, img_w, _ = frame.shape
            x_min, y_min = img_w, img_h
            x_max, y_max = 0, 0
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Add a margin to the bounding box.
            margin = 50
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img_w, x_max + margin)
            y_max = min(img_h, y_max + margin)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            roi = frame[y_min:y_max, x_min:x_max]
        else:
            # Reset if no hand is detected.
            streak_count = 0
            prev_predicted_index = -1
            prev_sent_mqtt = "-"
            if not headless:
                cv2.putText(frame, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                cv2.imshow("Sign Language Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # Preprocess ROI and run inference.
        input_data = preprocess(roi)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.squeeze(output_data)
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]

        # Apply penalty if necessary.
        current_sign = class_names[predicted_index]
        if current_sign in penalty_classes:
            confidence_adjusted = confidence - penalty_classes[current_sign]
        else:
            confidence_adjusted = confidence

        threshold = 0.9
        if confidence_adjusted > threshold:
            print(f"Detected {current_sign}: {confidence_adjusted:.2f}")

            # Update streak count.
            if predicted_index == prev_predicted_index:
                streak_count += 1
            else:
                streak_count = 1
                prev_predicted_index = predicted_index

            # Confirm detection after a number of consecutive frames.
            if streak_count >= min_streak and current_sign != prev_sent_mqtt:
                streak_count = 0
                text = f"{current_sign}: {confidence_adjusted:.2f}"
                if not headless:
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mqtt_client.publish(MQTT_TOPIC, current_sign)
                tts_queue.put(current_sign)
                prev_sent_mqtt = current_sign
        else:
            streak_count = 0
            prev_predicted_index = -1
            prev_sent_mqtt = "-"

        if not headless:
            cv2.imshow("Sign Language Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()
    tts_queue.join()

main()