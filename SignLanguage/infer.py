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
def get_network_range(interface='wlan0'):
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
MODEL_PATH = "best-fp16.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

def preprocess_frame(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (input_width, input_height))
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)
    return input_data

def postprocess(predictions, conf_threshold=0.5, iou_threshold=0.5):
    """
    Convert raw YOLOv5 TFLite output to detections.
    predictions: (num_predictions, 36) array.
      - First 4 values: center_x, center_y, width, height (normalized to input size).
      - 5th value: objectness score.
      - Remaining 31 values: class scores.
    Returns final boxes (x1, y1, x2, y2), confidences, and class_ids.
    """
    # Separate components
    boxes = predictions[:, :4]       # [cx, cy, w, h]
    objectness = predictions[:, 4:5]   # [objectness]
    class_scores = predictions[:, 5:]  # [class scores]

    # Compute detection scores
    scores = objectness * class_scores  # element-wise multiplication, shape (num, 31)
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter by confidence threshold
    mask = confidences >= conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if boxes.shape[0] == 0:
        return [], [], []

    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # Prepare boxes for NMS: convert to [x, y, w, h]
    boxes_xywh = []
    for b in boxes_xyxy:
        x1, y1, x2, y2 = b
        boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    
    # Apply Non-Maximum Suppression using OpenCV
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        indices = []

    final_boxes = boxes_xyxy[indices]
    final_confidences = confidences[indices]
    final_class_ids = class_ids[indices]

    return final_boxes, final_confidences, final_class_ids

def run_inference(frame):
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.squeeze(output_data)  # Shape: (num_predictions, 36)
    boxes, confs, class_ids = postprocess(predictions, conf_threshold=0.5, iou_threshold=0.5)
    return boxes, confs, class_ids

# =============================
# Main Inference and MQTT Publishing
# =============================
def main():
    # Choose MQTT broker from network scan
    chosen_ip = choose_broker_ip()
    if not chosen_ip:
        print("No valid MQTT broker selected. Exiting.")
        return

    MQTT_BROKER = chosen_ip  # Selected IP as broker
    MQTT_PORT = 1883
    MQTT_TOPIC = "assistedge/sign_language"

    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print("Error connecting to MQTT broker:", e)
        return
    mqtt_client.loop_start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define class labels (update as needed)
    classes = ['N', 'a', 'b', 'c', 'ch', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'll', 
               'm', 'n', 'o', 'otro', 'p', 'q', 'r', 'rr', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    prev_pred_class = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                continue

            frame = cv2.flip(frame, 1)
            # Run inference and get detections
            boxes, confs, class_ids = run_inference(frame)
            if len(boxes) > 0:
                # Choose the detection with the highest confidence
                max_idx = np.argmax(confs)
                box = boxes[max_idx]
                pred_index = class_ids[max_idx]
                pred_class = classes[pred_index] if pred_index < len(classes) else "unknown"
                confidence = confs[max_idx]

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{pred_class}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Publish to MQTT if confidence is high and prediction has changed
                if confidence > 0.55 and pred_class != prev_pred_class:
                    mqtt_client.publish(MQTT_TOPIC, pred_class)
                    prev_pred_class = pred_class
            else:
                pred_class = "unknown"
                cv2.putText(frame, pred_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

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
