import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate, Conv2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.initializers import he_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from globals import *

# Adjustable parameters
dataset_dir = 'EmoSet-118K'
test_size = 0.2
epochs = 10
batch_size = 64

# Loads and pre-processes image data
def load_dataset(data):
    images = []
    brightness_values = []
    colorfulness_values = []
    labels = []
    for entry in data[:50000]:
        emotion, image_path, annotation_path = entry
        if emotion in emotions:
            img_array = preprocess_image(os.path.join(dataset_dir, image_path))
            images.append(img_array)
            
            annotation_data = json.load(open(os.path.join(dataset_dir, annotation_path)))
            
            brightness = annotation_data.get('brightness', None)
            brightness_values.append(brightness if brightness is not None else 0.0)
            
            colorfulness = annotation_data.get('colorfulness', None)
            colorfulness_values.append(colorfulness if colorfulness is not None else 0.0)
            
            labels.append(emotions.index(emotion))
    return np.array(images), np.array(brightness_values), np.array(colorfulness_values), np.array(labels)

def main():
    train_data = json.load(open(os.path.join(dataset_dir, 'train.json')))
    val_data = json.load(open(os.path.join(dataset_dir, 'val.json')))

    X_train, brightness_train, colorfulness_train, y_train = load_dataset(train_data)
    X_val, brightness_val, colorfulness_val, y_val = load_dataset(val_data)

    # Convert labels to one-hot encoded format
    y_train = to_categorical(y_train, num_classes=len(emotions))
    y_val = to_categorical(y_val, num_classes=len(emotions))

    X_train, X_test, brightness_train, brightness_test, colorfulness_train, colorfulness_test, y_train, y_test = train_test_split(X_train, brightness_train, colorfulness_train, y_train, test_size=test_size, random_state=42)

    # Model Inputs
    image_input = Input(shape=(img_width, img_height, 3))
    brightness_input = Input(shape=(1,))
    colorfulness_input = Input(shape=(1,))

    # Model Architecture
    conv1 = Conv2D(32, (3, 3), activation='relu')(image_input)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    flatten = Flatten()(pool2)
    concatenated_inputs = Concatenate()([flatten, brightness_input, colorfulness_input])
    dense1 = Dense(512, activation='relu', kernel_initializer=he_normal())(concatenated_inputs)
    dropout1 = Dropout(0.5)(dense1)
    batch_norm1 = BatchNormalization()(dropout1)
    dense2 = Dense(256, activation='relu', kernel_initializer=he_normal())(batch_norm1)
    dropout2 = Dropout(0.5)(dense2)
    batch_norm2 = BatchNormalization()(dropout2)
    output = Dense(len(emotions), activation='softmax')(batch_norm2)

    model = Model(inputs=[image_input, brightness_input, colorfulness_input], outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Call backs to mitigate overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Display Model Architecture as Image File
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    img = plt.imread('model_plot.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Fit and save model
    history = model.fit([X_train, brightness_train, colorfulness_train], y_train, epochs=epochs, batch_size=batch_size, 
            validation_data=([X_val, brightness_val, colorfulness_val], y_val), 
            callbacks=[early_stopping, reduce_lr])

    model.save(img_model)
    print("Model saved successfully.")

    # Loss and Accuracy Values
    loss, accuracy = model.evaluate([X_test, brightness_test, colorfulness_test], y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Training and Validation Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Training and Validation Accuracy Curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion Matrix
    y_pred = model.predict([X_test, brightness_test, colorfulness_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Precision, Recall, and F1-score
    print(classification_report(np.argmax(y_test, axis=1), y_pred_classes))

if __name__ == "__main__":
    main()