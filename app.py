import uvicorn
import comment

import numpy as np
import tensorflow as tf

import tensorflow_text as text  # Used by BERT model in the ".h5" file
import tensorflow_hub as hub
from fastapi import FastAPI
from comment import Comment

app = FastAPI()

comment_classifier = tf.keras.models.load_model("comment_classifier.h5",
                                                custom_objects={'KerasLayer': hub.KerasLayer}
                                                )
CLASSES = ["toxic",
           "severe_toxic",
           "obscene",
           "threat",
           "insult",
           "identity_hate",
           "safe"]


# Index page
@app.get("/")
def home():
    return {"Message": "Welcome to Triumph's Deployment of his Comment Classifier Model."}


# Prediction page
@app.post("/predict")
def predict_class(raw_comment: Comment):
    raw_comment = raw_comment.dict()
    raw_comment = raw_comment["raw_comment"]

    clean_comment = comment.clean_comments(raw_comment)  # Clean text input

    prediction = comment_classifier.predict([clean_comment])
    classification = np.where(prediction > 0.5, 1, 0)

    result = dict()

    for i, j in zip(CLASSES, classification[0]):
        if j == 1:
            result[i] = "Yes"
        elif j == 0:
            result[i] = "No"

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# run this in your terminal
# uvicorn app:app --reload
