# 🎤 Voice Verification System using HMM and MFCC

## 📌 Objective
The main objective of this project is to **verify a speaker's identity** based on their voice using **Hidden Markov Models (HMM)** and **Mel-Frequency Cepstral Coefficients (MFCC)**. The system checks whether a given test audio sample matches a previously trained speaker's voice and grants or denies access accordingly.

---

## 🛠️ Tools and Technologies Used

- **Python** – Programming language
- **Librosa** – For audio loading and MFCC feature extraction
- **SoundFile** – For reading and writing audio files
- **hmmlearn** – For training and using Gaussian HMM models
- **joblib** – For saving and loading trained HMM models
- **Matplotlib** – For waveform and MFCC visualization
- **NumPy** – For efficient numerical computations

---

## ✨ Features

- Extracts MFCC features from audio samples.
- Verifies voice by computing log-likelihood scores using HMM.
- Grants or denies access based on score threshold.
- Visualizes waveform and MFCCs for better understanding of audio signals.

---

## 🧩 Key Steps

1. **Feature Extraction:**
   - MFCC features are extracted from audio using `librosa.feature.mfcc()`.

2. **Model Training (not shown in code):**
   - A Gaussian HMM is trained on the speaker's voice samples.
   - The trained model is saved using `joblib`.

3. **Voice Prediction:**
   - The test audio is passed through the same MFCC extraction process.
   - The trained model computes a log-likelihood score for the test voice.
   - If the score exceeds a defined threshold (e.g., -9000), access is granted.

4. **Visualization:**
   - Displays waveform and MFCC plots of test and reference audios for analysis and comparison.

---

## 🖼️ Output

- ✅ If test voice is similar: `🔓 Access Granted: This is your voice.`
- ❌ If test voice is different: `❌ Access Denied: Voice does not match.`
- 📊 Visual plots of waveform and MFCCs for comparison between audios.

---

## 📌 Conclusion

This voice verification system demonstrates a simple but effective approach to biometric authentication using voice. By leveraging MFCC for feature extraction and HMM for modeling temporal audio patterns, it provides a foundation for building secure voice-based access control systems.

---

## 📁 Example Files Used
- `TEST_CORRECT_1.wav` – Audio of the correct speaker.
- `TEST_NOTCORRECT_1.wav` – Audio from a different person.
- `test_audio.wav` – Used for visualization.

---

## 🚀 Future Improvements
- Implement real-time microphone input for live verification.
- Use advanced models like GMM-HMM or deep learning architectures for better accuracy.
- Build a user interface for non-technical use.

---

## 🔒 Disclaimer
This is a basic educational implementation and should not be used in critical security systems without further enhancements.

