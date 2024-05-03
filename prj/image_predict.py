import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from skimage import color
from globals import *

def preprocess_image_with_features(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = preprocess_image(image_path)
    
    brightness = np.mean(color.rgb2gray(img))
    colorfulness = np.mean(np.std(img, axis=(0, 1))) / 255.0
    
    return img_array, brightness, colorfulness

def use_model_to_predict(image_path):
    model = load_model(img_model)

    img_array, brightness, colorfulness = preprocess_image_with_features(image_path)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict([img_array, np.array([[brightness]]), np.array([[colorfulness]])])

    # Convert numeric result to emotion string
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = emotions[predicted_emotion_index]

    # Reduce number of emotions to more generalized ones
    if predicted_emotion == "amusement" or predicted_emotion == "excitement":
        predicted_emotion = "happy"
    elif predicted_emotion == "contentment" or predicted_emotion == "awe":
        predicted_emotion = "relax"
    elif predicted_emotion == "disgust" or predicted_emotion == "fear":
        predicted_emotion = "anger"
    elif predicted_emotion == "sadness":
        predicted_emotion = "sad"

    return predicted_emotion

def predict_emotion():
    image_path = input("Enter the path to the image: ")

    if not os.path.exists(image_path):
        print("Error: Image path does not exist.")
        return

    predicted_emotion = use_model_to_predict(image_path)
    print("Predicted Emotion:", predicted_emotion)

    return predict_emotion
