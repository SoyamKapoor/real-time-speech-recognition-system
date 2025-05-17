import os
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt
from hmmlearn import hmm

# ---------------------------
# Step 1: Extract MFCCs
# ---------------------------

def extract_mfcc_from_folder(folder_path):
    mfcc_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            y, sr = librosa.load(file_path, sr=16000)
            y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
            mfcc_list.append(mfcc)
    return mfcc_list

your_mfccs = extract_mfcc_from_folder("Dataset/soyam_voice")
others_mfccs = extract_mfcc_from_folder("Dataset/other_voice")

# Save extracted features (optional)
with open("your_mfcc.pkl", "wb") as f: pickle.dump(your_mfccs, f)
with open("others_mfcc.pkl", "wb") as f: pickle.dump(others_mfccs, f)

# ---------------------------
# Step 2: Train and Select Best HMM
# ---------------------------

def train_hmm(mfcc_data, n_components=5):
    X = np.concatenate(mfcc_data)
    lengths = [len(seq) for seq in mfcc_data]
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000, tol=1e-4)
    model.fit(X, lengths)
    return model

def get_avg_score(model, mfcc_data):
    return np.mean([model.score(seq) for seq in mfcc_data])

best_model = None
best_gap = float('-inf')
best_scores = {}

print("\n🔁 Trying different HMMs:")
for n in range(4, 11):  # try components from 4 to 10
    print(f"Training HMM with {n} components...")
    model = train_hmm(your_mfccs, n_components=n)
    your_score = get_avg_score(model, your_mfccs)
    others_score = get_avg_score(model, others_mfccs)
    gap = your_score - others_score
    print(f"✅ Your score: {your_score:.2f}, ❌ Others score: {others_score:.2f}, Gap: {gap:.2f}")

    if gap > best_gap:
        best_gap = gap
        best_model = model
        best_scores = {"your_score": your_score, "others_score": others_score}

# Save best model
with open("best_hmm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n🎯 Best model selected!")
print(f"🔐 Your voice avg log-likelihood: {best_scores['your_score']:.2f}")
print(f"❌ Others' voice avg log-likelihood: {best_scores['others_score']:.2f}")

# ---------------------------
# Step 3: Visualize Score Distributions
# ---------------------------

your_scores = [best_model.score(seq) for seq in your_mfccs]
others_scores = [best_model.score(seq) for seq in others_mfccs]

plt.hist(your_scores, bins=10, alpha=0.6, label='Your Voice')
plt.hist(others_scores, bins=10, alpha=0.6, label='Others')
plt.axvline(np.mean(your_scores), color='blue', linestyle='dashed', label='Your Avg')
plt.axvline(np.mean(others_scores), color='red', linestyle='dashed', label='Others Avg')
plt.title("Voice Likelihood Score Distribution")
plt.xlabel("Log-Likelihood Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ---------------------------
# Step 4: Test Prediction
# ---------------------------

def predict_voice(file_path, model, threshold):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    score = model.score(mfcc)

    print(f"\n📣 Test voice score: {score:.2f}")
    if score >= threshold:
        print("✅ This is likely YOUR voice.")
    else:
        print("❌ This is NOT your voice.")

# Calculate threshold
threshold = (best_scores['your_score'] + best_scores['others_score']) / 2

# Example usage
predict_voice("test_audio.wav", best_model, threshold)
