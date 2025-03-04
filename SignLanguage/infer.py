import netifaces
import ipaddress
import nmap
import threading
import cv2
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
import time

# =============================
# Network Discovery Functions
# =============================
#wlan0
def get_network_range(interface='wlan0'):
    """
    Retrieve the local network range based on the IP and netmask of the specified interface.
    Returns the network range in CIDR notation (e.g., "172.20.10.0/28").
    """
    try:
        addrs = netifaces.ifaddresses(interface)
        inet_info = addrs[netifaces.AF_INET][0]
        ip_addr = inet_info['addr']
        netmask = inet_info['netmask']
        # Create an IPv4 network object with strict=False so host bits can be non-zero
        network = ipaddress.IPv4Network(f"{ip_addr}/{netmask}", strict=False)
        return str(network)
    except Exception as e:
        print("Error retrieving network range:", e)
        return None

def scan_network(network_range):
    """
    Scans the specified network range and returns a list of tuples (hostname, ip).
    """
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
    """
    Discover devices on the local network and prompt the user to select one.
    Returns the selected device's IP address.
    """
    network_range = get_network_range('en0')
    if not network_range:
        print("Could not determine network range. Ensure 'wlan0' is active.")
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
# TensorFlow Lite Inference Setup
# =============================

MODEL_PATH = "asl.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (input_width, input_height))
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)
    return input_data

def run_inference(frame):
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# =============================
# Main Inference and MQTT Publishing
# =============================

def main():
    # Prompt to choose MQTT_BROKER IP from discovered devices.
    chosen_ip = choose_broker_ip()
    if not chosen_ip:
        print("No valid MQTT broker selected. Exiting.")
        return

    MQTT_BROKER = chosen_ip  # Use the selected IP as broker
    MQTT_PORT = 1883
    MQTT_TOPIC = "assistedge/sign_language"

    # Setup MQTT client
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print("Error connecting to MQTT broker:", e)
        return
    mqtt_client.loop_start()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define class labels (update as per your model training)
    classes = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Delete', 'Nothing'
    ]
    
    prev_pred_class = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                continue

            frame = cv2.flip(frame, 1)
            output_data = run_inference(frame)
            pred_index = np.argmax(output_data)
            pred_class = classes[pred_index] if pred_index < len(classes) else "unknown"
            confidence = np.max(output_data)
            
            # Only publish if confidence is high and new prediction is different from the previous one.
            if confidence > 0.78 and pred_class != prev_pred_class:
                print(f"Predicting: {pred_class} with {confidence} confidence")
                display_text = f"{pred_class}: {confidence:.2f}"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Sign Language Recognition", frame)

                mqtt_client.publish(MQTT_TOPIC, pred_class)
                prev_pred_class = pred_class
            else:
                # Optionally, update display without publishing if needed.
                display_text = f"{pred_class}: {confidence:.2f}"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Sign Language Recognition", frame)
            
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()
