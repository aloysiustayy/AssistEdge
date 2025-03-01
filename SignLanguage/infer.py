import cv2
import numpy as np
import tensorflow as tf

# Path to your TFLite model file
MODEL_PATH = "asl.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Retrieve expected input shape from the model.
# Assume shape format is [batch, height, width, channels]
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

def preprocess_frame(frame):
    """
    Resize the frame to match the model's expected input size,
    normalize pixel values, and add a batch dimension.
    """
    # Resize the frame
    resized_frame = cv2.resize(frame, (input_width, input_height))
    # Normalize pixel values to [0, 1] if model expects float32 input
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    # Add batch dimension
    input_data = np.expand_dims(normalized_frame, axis=0)
    return input_data

def run_inference(frame):
    """
    Preprocess a frame, run inference, and return the output.
    """
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def main():
    # Open the default webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define your class names (update as per your model)
    classes = ['A', 'B',' C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',' U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Delete', 'Nothing']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Optionally, flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Run inference on the current frame
        output_data = run_inference(frame)

        # print("Output shape:", output_data.shape)
        # print("Output data:", output_data)

        # For demonstration, assume output_data is a probability vector for each class
        pred_index = np.argmax(output_data)
        pred_class = classes[pred_index]
        confidence = np.max(output_data)

        

        # Overlay the predicted class and confidence on the frame
        display_text = f"{pred_class}: {confidence:.2f}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Sign Language Recognition", frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
