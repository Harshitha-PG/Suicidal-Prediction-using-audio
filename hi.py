import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set a fixed shape for all spectrograms
FIXED_SHAPE = (128, 400)

# Function to extract Mel spectrogram features and ensure a fixed shape
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Ensure fixed shape by padding or truncating
        if mel_spectrogram.shape[1] < FIXED_SHAPE[1]:
            pad_width = FIXED_SHAPE[1] - mel_spectrogram.shape[1]
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :FIXED_SHAPE[1]]
        
        return mel_spectrogram
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

# Function to load dataset and extract features
def load_dataset(folder_path):
    features = []
    labels = []
    
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                mel_spectrogram = extract_features(file_path)
                if mel_spectrogram is not None:
                    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension
                    features.append(mel_spectrogram)
                    # Assuming 'suicidal' in filename indicates positive class, otherwise negative
                    label = 1 if 'suicidal' in file_name.lower() else 0
                    labels.append(label)
    
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

# Function to build a CNN model
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main training and evaluation process
def main():
    # Define dataset path and model save path directly in the code
    folder_path = r'C:\Users\pghar\OneDrive\Desktop\spltopics\Audio_Speech_Actors_01-24\Actor_01'
    model_save_path = 'cnn_suicidal_prediction_model.h5'
    
    print("Loading dataset...")
    X, y = load_dataset(folder_path)
    print(f'Dataset loaded with {len(X)} samples.')

    if len(X) == 0:
        print("No valid data found. Please check the dataset path.")
        return

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Building and training the model
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    
    print("Training model...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                        validation_split=0.2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
                        verbose=1)
    
    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
