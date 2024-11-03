from flask import Flask, render_template, request, redirect, url_for
from pydub import AudioSegment
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
# Load model and encoder
model = load_model('best_model.keras')
encoder = joblib.load('label_encoder.pkl')

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")

# Function to extract MFCCs from WAV file
def extract_mfcc(file_path, n_mfcc=40, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Pad or truncate to match max_pad_len
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    return mfcc

# Function to predict speaker from MP3 file
def predict_speaker(mp3_path):
    # Convert MP3 to WAV
    wav_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    convert_mp3_to_wav(mp3_path, wav_file)
    
    # Extract MFCCs and reshape for model
    mfccs = extract_mfcc(wav_file)
    mfccs = mfccs[np.newaxis, ..., np.newaxis]

    # Predict speaker
    predictions = model.predict(mfccs)
    predicted_label = np.argmax(predictions, axis=1)[0]
    
    # Get speaker name
    speaker_name = encoder.inverse_transform([predicted_label])[0]
    return speaker_name

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save uploaded file
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Ensure the file is saved before prediction

        # Predict speaker
        speaker_name = predict_speaker(file_path)
        
        # Remove the uploaded file after prediction
        os.remove(file_path)
        
        return render_template('index.html', prediction=speaker_name)
    
if __name__ == "__main__":
    app.run()  # Runs on localhost at port 5000 by default

