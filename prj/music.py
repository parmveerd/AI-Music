import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import librosa
import librosa.display
from music21 import stream, note
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical
from PIL import Image
from pathlib import Path
from globals import *
from image_predict import predict_emotion


def create_model(X_train_reshaped, y_train):
    # Build the RNN model
    model = Sequential([
        Dropout(0.5),
        LSTM(512, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),
        Dense(256),
        Dense(256),
        LSTM(256, return_sequences=True),
        Dense(128),
        LSTM(128),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax') 
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_reshaped, y_train, epochs=75, batch_size=32, validation_split=0.2)

    model.save(music_model)

    return model, history

def generate_mfcc_features(model, initial_features, num_steps=50):
    
    generated_features = [initial_features]

    input_array = np.expand_dims(generated_features[-1], axis=0)

    for _ in range(num_steps):
        # Resize input
        input_array = input_array.reshape((1, -1, 1))
        
        # Standardize the pattern
        input_array = input_array / 80

        # Predict with model and appent
        prediction = model.predict(input_array)
        generated_features.append(prediction)

    return generated_features



def audio_features(generated_features):
    # Sampling rate
    sr = 22050

    # Transpose the feature array
    feature_transposed = np.array(generated_features[0]).T

    # Invert the MFCC features to audio
    audio_reconstructed = librosa.feature.inverse.mfcc_to_audio(feature_transposed)

    # Plot the spectrogram of the reconstructed audio
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_reconstructed, sr=sr)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Reconstructed Audio from MFCC')
    plt.show()



def generate_music(generated_features):
    mfcc_values = generated_features[0]

    midi_notes = np.interp(mfcc_values, (mfcc_values.min(), mfcc_values.max()), (60, 72))

    # Create a simple melody using the mapped MIDI notes
    melody = [(note, random.uniform(0.5, 1)) for note in midi_notes]

    # Create a stream object to represent the melody
    melody_stream = stream.Stream()

    # Add notes to the melody stream based on the generated melody
    for midi_note, duration in melody:
        midi_note_scalar = midi_note.item()
        note_obj = note.Note(midi=midi_note_scalar)
        note_obj.duration.quarterLength = duration
        melody_stream.append(note_obj)

    # Save the melody stream as a MIDI file
    midi_filename = "generated_melody.mid"
    melody_stream.write("midi", fp=midi_filename)


def main():
    # Load the data
    df = pd.read_csv('Acoustic Features.csv')

    # Split the data into MFCC and emotion labels
    X = df.drop(columns=['Class']) 
    X = X.iloc[:, 4:17]
    y = df['Class']

    # Convert labels to one-hot encoded format
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    # Split the data into training and testing sets (4:1 ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape input data to include sequence length dimension
    X_train_reshaped = X_train_scaled[..., np.newaxis]
    X_test_reshaped = X_test_scaled[..., np.newaxis]

    # Create RNN model
    model, history = create_model(X_train_reshaped, y_train)

    # Predict with model
    y_pred = model.predict(X_test_reshaped)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes))

    # Print confusion matrix
    conf = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    emotion = predict_emotion()

    # Seed index
    if (emotion == "angry"):
        index = random.randint(0, 15)
    elif (emotion == "happy"):
        index = random.randint(20, 30)
    elif (emotion == "relax"):
        index = random.randint(40, 50)
    else:
        index = random.randint(69, 79)
    
    initial_features = X_test_reshaped[index]

    # Generate new MFCC features
    generated_features = generate_mfcc_features(model, initial_features)

    # Generate music
    audio_features(generated_features)
    generate_music(generated_features)


if __name__ == "__main__":
    main()