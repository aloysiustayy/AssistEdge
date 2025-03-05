import speech_recognition as sr
import whisper
import numpy as np
import sys

class WhisperRecognizer:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
    
    def recognize(self, audio_data):
        # Convert the audio data to a NumPy array
        audio = np.frombuffer(audio_data.get_raw_data(), np.int16).astype(np.float32) / 32768.0

        # Use Whisper to transcribe the audio
        result = self.model.transcribe(audio, fp16=False)
        return result["text"]

def find_microphone(mic_name):
    """Find a microphone by name."""
    mic_list = sr.Microphone.list_microphone_names()
    
    # Print all available microphones
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
    # Implement the text processing here
    print(f"Processed transcription: {text}")

def main():
    mic_name = "JOYACCESS: USB Audio"
    
    # Find the microphone by name
    mic_index = find_microphone(mic_name)
    
    if mic_index is None:
        print(f"Microphone '{mic_name}' not found. Please check the available devices listed above.")
        sys.exit(1)

    try:
        mic = sr.Microphone(device_index=mic_index)
        r = sr.Recognizer()
        whisper_recognizer = WhisperRecognizer(model_size="base")

        print("\nListening... Press Ctrl+C to stop.")

        with mic as source:
            r.adjust_for_ambient_noise(source)
            while True:
                print("Listening...")
                audio = r.listen(source)
                
                try:
                    text = r.recognize_google(audio)
                    print(f"You said: {text}")
                    
                    # Process the text using the dedicated callback function
                    process_transcription(text)
                    
                except Exception as e:
                    print(f"Could not recognize speech; {e}")

    except KeyboardInterrupt:
        print("\nTerminating the program.")
        sys.exit(0)

if __name__ == "__main__":
    main()
