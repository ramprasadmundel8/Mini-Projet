
# 🧠 Sign Language Interpreter Using Deep Learning

This project is a **vision-based communication system** for people who are deaf or mute. It interprets sign language gestures captured from hand landmarks using **MediaPipe**, and classifies them using **deep learning models** like CNN, RNN, and LSTM.

---

## 📌 Problem Statement

Sign language is the primary mode of communication for millions of people with speech and hearing disabilities. However, its widespread adoption is limited due to a lack of understanding among the general public. This project bridges the gap by converting hand gestures into machine-understandable formats and classifying them to recognize signs.

---

## 🎯 Objectives

- Build an AI model that can understand and classify sign language gestures.
- Extract hand features using **MediaPipe**.
- Compare different deep learning models:
  - CNN
  - CNN + RNN
  - CNN + RNN + LSTM
- Improve model accuracy using **data augmentation** and hybrid architectures.

---

## 🧰 Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe** (for hand landmark detection)
- **TensorFlow
- **Matplotlib**
- **Scikit-learn**

---

## 🗂️ Project Structure

.
├── Data/                         # Dataset containing sign language images
├── Classifier Mediapipe+CNN.py                # CNN-based model
├── Classifier Mediapipe+CNN+RNN.py            # CNN + RNN hybrid model
├── Classifier Mediapipe+CNN+RNN2..py          # CNN + Bidirectional LSTM with callbacks
├── DL_Models/                   # Trained model files (.h5)
└── README.md                    # Project documentation

Future Improvements
Integrate with webcam for real-time gesture prediction.

Extend to dynamic gestures using video sequences.

Deploy using Streamlit or Flask for live demos.


