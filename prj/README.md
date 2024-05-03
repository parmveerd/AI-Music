If this is your first time running the model, call:

    python create_models.py

To run program and generate music:

    python main.py

Ensure that you have all the imports installed.

Code Structure:

create_models.py
    - Generates models for image_model and music_model

music.py
    - Generates music from image

image_predict.py
    - Takes image as input and outputs predicted emotion

globals.py
    - Stores global functions/variables used throughout project

Data set Structure:
Ensure EmoSet-118K folder and Acoustic Features.csv are in same location as code.

EmoSet-118K/
    annotation/
        amusement/
            *.json
        anger/
        awe/
        contentment/
        disgust/
        excitement/
        fear/
        sadness/
    image/
        amusement/
            *.jpg
        anger/
        awe/
        contentment/
        disgust/
        excitement/
        fear/
        sadness/
    info.json
    test.json
    train.json
    val.json
Acoustic Features.csv

Note: Due to the size of the data sets, we omitted it from this submission. To download the data sets, visit these links:

https://vcc.tech/EmoSet
https://www.kaggle.com/datasets/blaler/turkish-music-emotion-dataset

Self-Evaluation:
We believe that our project accomplished what we wanted to in the proposal. However, there are some problems we faced that made the results less than we expected. Having to reduce the range of emotions we could cover due to the limited music data set was disappointing. However, the outputted music is amazing to hear and reflects the inputted image well.

