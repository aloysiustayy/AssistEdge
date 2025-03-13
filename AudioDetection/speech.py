import joblib
import librosa
import numpy as np

# Load the trained model and scaler
model = joblib.load('speech_recognition_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to extract MFCC features
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Function to recognize the spoken word
def recognize_speech(file_path):
    mfccs = extract_mfcc(file_path)
    mfcc_scaled = scaler.transform([mfccs])
    prediction = model.predict(mfcc_scaled)
    return prediction[0]

# Example usage on Raspberry Pi
result = recognize_speech('test.wav')
print(f"Predicted Word: {result}")
