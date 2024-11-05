import streamlit as st
from pydub import AudioSegment
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import os

# Load the model and encoder
model = load_model('best_model.keras')
encoder = joblib.load('label_encoder.pkl')

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    audio.export(wav_file, format="wav")

# Function to extract MFCCs from WAV file
def extract_mfcc(file_path, n_mfcc=40, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc)
    
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    return mfcc

# Function to predict speaker from MP3 file
def predict_speaker(mp3_path):
    wav_file = 'temp_audio.wav'
    convert_mp3_to_wav(mp3_path, wav_file)
    
    # Extract MFCCs and reshape for model
    mfccs = extract_mfcc(wav_file)
    mfccs = mfccs[np.newaxis, ..., np.newaxis]

    # Predict speaker
    predictions = model.predict(mfccs)
    predicted_label = np.argmax(predictions, axis=1)[0]
    
    # Get speaker name
    speaker_name = encoder.inverse_transform([predicted_label])[0]
    
    # Clean up temporary WAV file
    os.remove(wav_file)
    
    return speaker_name

# Streamlit UI
st.title("Speaker Recognition App")
st.write("Upload an MP3 file to identify the speaker.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = 'uploaded_audio.mp3'
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the file and predict speaker
    speaker_name = predict_speaker(file_path)
    
    # Display result
    st.success(f"Predicted Speaker: {speaker_name}")
    
    # Clean up the uploaded file
    os.remove(file_path)
