import os
import numpy as np
import librosa
import pickle


# ---------------------------
# 1. Extract MFCC from files
# ---------------------------

def extract_mfcc_from_folder(folder_path):
    mfcc_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            y, sr = librosa.load(os.path.join(folder_path, file), sr=16000)
            y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
            mfcc_list.append(mfcc)
    return mfcc_list

your_mfccs = extract_mfcc_from_folder("Dataset/soyam_voice")
others_mfccs = extract_mfcc_from_folder("Dataset/other_voice")

# Save for reuse
with open("your_mfcc.pkl", "wb") as f: pickle.dump(your_mfccs, f)
with open("others_mfcc.pkl", "wb") as f: pickle.dump(others_mfccs, f)