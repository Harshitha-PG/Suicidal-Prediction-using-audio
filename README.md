# 🎧 Suicidal Prediction Using Audio with CNN

This project builds a deep learning model that predicts **suicidal tendencies** from audio recordings using **Convolutional Neural Networks (CNN)**. The system analyzes vocal emotion and acoustic features to estimate the likelihood of suicidality and presents the result as a **percentage bar graph**.

---

## 📁 Dataset Used: RAVDESS

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)  
- Public dataset from Kaggle: [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)  
- Contains 24 professional actors speaking two lexically-matched statements in a range of emotions (calm, happy, sad, angry, fearful, etc.)  
- Only audio files are used for this project

---

## 🧠 Project Overview

### 🔹 Objective
- Predict suicidal tendencies based on vocal emotion and speech features
- Assist in **early mental health detection** using voice alone
- Output a **suicidality percentage score** visualized in a **bar chart**

---

## 🧪 Methodology

### 🎙️ Audio Preprocessing
- Convert `.wav` to Mel-spectrograms using `librosa`
- Extract key features: pitch, tone, MFCCs (Mel Frequency Cepstral Coefficients)
- Normalize and prepare images for CNN input

### 🧠 CNN Model Architecture
- 2D Convolutional layers with ReLU
- MaxPooling and Dropout for regularization
- Dense layers for prediction
- Output: Emotional category → mapped to suicidality score

### 📊 Suicidal Scoring
- Emotion Prediction → Suicidal Weighting:
  - **Sad / Fearful** → Higher score
  - **Neutral / Calm** → Lower score

---

## 🔧 Tech Stack

- **Python** – Core programming language used for the entire pipeline
- **Librosa** – For audio preprocessing, feature extraction (e.g., MFCCs, pitch, tempo)
- **TensorFlow / Keras** – For building and training the Convolutional Neural Network (CNN)
- **Matplotlib / Seaborn** – For data visualization and plotting bar graphs of suicidality
- **NumPy & Pandas** – For data manipulation, preprocessing, and handling audio metadata
- **Scikit-learn (Sklearn)** – For encoding, splitting datasets, evaluating model performance

---

## 🚀 Getting Started

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Preprocess audio into spectrograms
python preprocess_audio.py

# Step 3: Train the model
python train_model.py

# Step 4: Predict suicidality from a test audio file
python predict.py --audio path/to/audio.wav
```
---
##
📊 Visualization

The following graph is a sample output that demonstrates how the model visualizes the **suicidal tendency probability** based on the audio input:

![image](https://github.com/user-attachments/assets/ffbafd1c-6dda-4139-a901-4d1479872ff6)


- The x-axis shows the classification categories: **Non-Suicidal** and **Suicidal**  
- The y-axis represents the **predicted probability**  
- In this case, the model confidently predicts the person as **non-suicidal**

---

## 📍 Project Scope & Applications

This project lays a strong foundation for **mental health detection using voice analysis**, and it can be extended in multiple directions:

### 🔮 Future Scope
- 🔄 **Multimodal Expansion**: Combine audio + facial expression from videos for more accurate emotion + mental health detection
- 💬 **Live Monitoring**: Integrate into real-time communication platforms for background wellness analysis
- 🧠 **Therapy Support Tools**: Aid therapists by providing emotion/suicidality insights from recorded sessions
- 🧪 **Mental Health Triage**: Automate initial screening in mental health apps or clinics
- 🛠 **HR & Interview Analytics**: Assist recruiters by analyzing emotional stability during interviews
- 📱 **Mobile App Integration**: Convert this into a mobile-first mental health assistant

> This system is **non-invasive**, privacy-preserving, and could greatly assist in **early mental health intervention** when combined with professional care.

---

## ⚠️ Ethical Use

- This project should only be used in settings with **consent and transparency**
- It should complement — not replace — **clinical evaluation**
- Never use for profiling or discrimination; always consult mental health professionals



