
# 🧠 Sign Language Interpreter Using Deep Learning
This project is a **vision-based communication system** for people who are deaf or mute. It interprets sign language gestures captured from hand landmarks using **MediaPipe**, and classifies them using **deep learning models** like CNN, RNN, and LSTM.

## 📌 Problem Statement
Sign language is the primary mode of communication for millions of people with speech and hearing disabilities. However, its widespread adoption is limited due to a lack of understanding among the general public. This project bridges the gap by converting hand gestures into machine-understandable formats and classifying them to recognize signs.


## 🎯 Objectives
- Build an AI model that can understand and classify sign language gestures.
- Extract hand features using **MediaPipe**.
- Compare different deep learning models:
  - CNN
  - CNN + RNN
  - CNN + RNN + LSTM
- Improve model accuracy using **data augmentation** and hybrid architectures.

## 🧰 Technologies Used
- **Python**
- **OpenCV**
- **MediaPipe** (for hand landmark detection)
- **TensorFlow
- **Matplotlib**
- **Scikit-learn**

## 🗂️ Project Structure
├── Data/                         # Dataset containing sign language images
├── Classifier Mediapipe+CNN.py                # CNN-based model
├── Classifier Mediapipe+CNN+RNN.py            # CNN + RNN hybrid model
├── Classifier Mediapipe+CNN+RNN2..py          # CNN + Bidirectional LSTM with callbacks
├── DL_Models/                   # Trained model files (.h5)
└── README.md                    # Project documentation

## 🧪 Models & Accuracy Comparison

| Model                     | Description                         | Accuracy (Approx.) |
|--------------------------|-------------------------------------|---------------------|
| **CNN**                  | Basic Conv1D-based model             | 93.45%              |
| **CNN + RNN**            | CNN with RNN for temporal modeling   | Higher than CNN     |
| **CNN + Bidirectional LSTM** | Best-performing hybrid model       | Best (~95%+)        |

Each model was trained on 24 sign classes (A–X, excluding J and Z due to motion).

## 🚀 How to Run
1. **Clone the repository**
git clone https://github.com/ramprasadmundel8/Mini-Projet.git
cd Mini-Projet
Install dependencies

pip install -r requirements.txt
Prepare the dataset
Add gesture image folders into the Data/ directory.
Each folder should be named after a label/class (e.g., 0, 1, ..., 23).

Train the model
python Classifier\ Mediapipe+CNN+RNN2..py

View results
Training and validation graphs will be plotted.
Trained model will be saved to DL_Models/.

📊 Sample Output Plots
Training accuracy/loss graphs are generated during model training for analysis and comparison.



Future Improvements
Integrate with webcam for real-time gesture prediction.
Extend to dynamic gestures using video sequences.
Deploy using Streamlit or Flask for live demos.


