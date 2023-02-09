# comment_classifier

## NLP project

**Data Source:** [Kaggle](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)

**Description:** This end-to-end Deep Learning project trained a 
model to receive a comment and give a response of if it falls into 
any of the predefined categories.

The categories are:

- "toxic"
- "severe_toxic"
- "obscene"
- "threat"
- "insult"
- "identity_hate"
- "safe"

**Approach:** This was done using [Bert](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4)
as a base model.

**Result:** The model attained an accuracy of **92.97%** after 5 epochs of training.

**Deployment:** Finally the model was deployed as an API endpoint using FastAPI.
