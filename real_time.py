import numpy as np              # For numerical operations
import librosa                 # For audio loading and MFCC extraction
import soundfile as sf         # Required by librosa to read/write audio
from hmmlearn.hmm import GaussianHMM  # For training and using HMM
import joblib                  # For saving and loading trained models
import librosa.display
import matplotlib.pyplot as plt
import os

def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def predict_voice_live(audio_path, model, threshold):
    mfcc = extract_mfcc(audio_path)
    score = model.score(mfcc)
    print(f"🎙️ Test Voice Log-Likelihood: {score:.2f}")
    if score > threshold:
        print("🔓 Access Granted: This is your voice.")
    else:
        print("❌ Access Denied: Voice does not match.")
best_model = joblib.load("best_hmm_model.pkl")
predict_voice_live("TEST_CORRECT_1.wav", best_model, threshold=-9000)


def visualize_audio(file_path, title):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    plt.figure(figsize=(12, 6))

    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform of {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot MFCCs
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCCs of {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")

    plt.tight_layout()
    plt.show()

# Run for each audio sample
visualize_audio("test_audio.wav", "Test Audio")
visualize_audio("TEST_CORRECT_1.wav", "Your Voice (Correct)")
visualize_audio("TEST_NOTCORRECT_1.wav", "Other Voice (Incorrect)")