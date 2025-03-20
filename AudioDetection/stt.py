import speech_recognition as sr
import numpy as np
import joblib
import librosa
import paho.mqtt.client as mqtt
import argparse

parser = argparse.ArgumentParser(description="Run emotion detection with optional GUI")
parser.add_argument("--ip", type=str, default="192.168.1.100", help="Flask Server IP address")
args = parser.parse_args()

# Load the trained MFCC model and scaler
model = joblib.load('speech_recognition_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to extract MFCC features from raw audio data
def extract_mfcc_from_audio_data(audio_data, sampling_rate, n_mfcc=13):
    # Convert audio_data to numpy array and ensure proper data type
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Function to recognize the spoken word using MFCC model
def recognize_speech_from_audio_data(audio_data, sampling_rate):
    mfccs = extract_mfcc_from_audio_data(audio_data, sampling_rate)
    mfcc_scaled = scaler.transform([mfccs])
    prediction = model.predict(mfcc_scaled)
    return prediction[0]

def find_microphone(mic_name):
    """Find a microphone by name."""
    mic_list = sr.Microphone.list_microphone_names()

    print("Available microphones:")
    for i, name in enumerate(mic_list):
        print(f"{i}: {name}")
        
    for i, name in enumerate(mic_list):
        if mic_name in name:
            print("Microphone found!")
            return i
    return None

def process_transcription(text):
    """Process or handle the transcription text."""
    print(f"Processed transcription: {text}")

def main():

    MQTT_BROKER = "192.168.1.100"
    if args.ip:
        MQTT_BROKER = args.ip
    
    MQTT_PORT = 1883
    MQTT_TOPIC = "assistedge/speech"


    # Setup MQTT client
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print("Error connecting to MQTT broker:", e)
        return
    mqtt_client.loop_start()

    # This mic name is hard-coded, if using other mic, need to find out microphone name
    mic_name = "JOYACCESS: USB Audio"
    
    mic_index = find_microphone(mic_name)
    if mic_index is None:
        print(f"Microphone '{mic_name}' not found. Please check the available devices listed above.")
        exit(1)

    try:
        mic = sr.Microphone(device_index=mic_index)
        recognizer = sr.Recognizer()

        print("\nListening... Press Ctrl+C to stop.")

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            while True:
                print("Listening...")
                audio = recognizer.listen(source)
                
                audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
                sampling_rate = audio.sample_rate

                try:
                    predicted_word = recognize_speech_from_audio_data(audio_data, sampling_rate)
                    print(f"AssistEdge Predicted Word: {predicted_word}")

                    process_transcription(predicted_word)
                    mqtt_client.publish(MQTT_TOPIC, predicted_word)

                except Exception as e:
                    print(f"Could not process the speech; {e}")

    except KeyboardInterrupt:
        print("\nTerminating the program.")
        exit(0)

if __name__ == "__main__":
    main()
