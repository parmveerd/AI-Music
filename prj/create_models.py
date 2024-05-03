from image_model import main as image_main
from music import main as music_main

def create_model():
    # Call the main functions of both image_model.py and music_model.py
    image_main()
    music_main()

if __name__ == "__main__":
    create_model()