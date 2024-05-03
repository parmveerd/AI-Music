from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_model = "image_emotion_model.h5"
music_model ="music_model.h5"

img_width, img_height = 224, 224 # Size values that other models (ie: VGG) use
emotions = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    return img_array / 255.0  # Normalize pixel values