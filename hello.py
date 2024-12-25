import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set a fixed shape for Mel spectrograms
FIXED_SHAPE = (128, 128)

# Function to extract Mel spectrogram features
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if y is None or len(y) == 0:
            return None
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Pad or truncate to FIXED_SHAPE
        if mel_spectrogram.shape[1] < FIXED_SHAPE[1]:
            pad_width = FIXED_SHAPE[1] - mel_spectrogram.shape[1]
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :FIXED_SHAPE[1]]
        return mel_spectrogram
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Function to load dataset and extract features
def load_dataset(folder_path):
    features = []
    labels = []
    print(f"Scanning folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith('.wav'):
                file_path = os.path.join(root, file_name)
                mel_spectrogram = extract_features(file_path)
                if mel_spectrogram is not None:
                    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
                    features.append(mel_spectrogram)
                    label = 1 if 'suicidal' in file_name.lower() else 0
                    labels.append(label)
    print(f"Total files processed: {len(features)}")
    return np.array(features), np.array(labels)

# Function to create CNN model
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# Function to predict and visualize a single audio file input by the user
def predict_single_audio(model):
    audio_path = input("Enter the path of the audio file to predict: ")
    if not os.path.exists(audio_path) or not audio_path.lower().endswith('.wav'):
        print("Invalid file path or format. Please provide a valid '.wav' file.")
        return

    mel_spectrogram = extract_features(audio_path)
    if mel_spectrogram is None:
        print("Error processing the audio file.")
        return

    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    prob = model.predict(mel_spectrogram)[0][0]

    # Plot the prediction as a bar graph
    plt.figure(figsize=(6, 4))
    plt.bar(['Non-suicidal', 'Suicidal'], [1 - prob, prob], color=['green', 'red'])
    plt.ylabel('Probability')
    plt.title(f'Suicidal Prediction: {"Suicidal" if prob > 0.5 else "Non-suicidal"}')
    plt.ylim(0, 1)
    plt.show()

# Main function to train the model and make predictions
def main():
    folder_path =r"C:\Users\pghar\OneDrive\Desktop\spltopics\Audio_Speech_Actors_01-24\Actor_01"
    
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset(folder_path)
    
    if len(X) == 0:
        print("No valid data found. Please check the dataset path.")
        return

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = create_model(input_shape=(FIXED_SHAPE[0], FIXED_SHAPE[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    
    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
    accuracy = np.mean(y_pred == y_test) * 100
    print(f'Accuracy: {accuracy:.2f}%')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    # Predict on a single audio file input by the user
    predict_single_audio(model)

if __name__ == '__main__':
    main()
