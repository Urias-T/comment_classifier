import re
import string

from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Comment(BaseModel):
    raw_comment: str


STOPWORDS = stopwords.words("english")
WORDNET_LEMMATIZER = WordNetLemmatizer()


def clean_comments(comment):
    """
    Takes in texts and cleans them to allow for better model performance.This done by
    changing to lowercase, removing links, removing punctuations and other steps.

    Args:
        comment (str): The comment to be cleaned.

    Returns:
        clean_comment (str): The comment which has been cleaned.
    """

    comment = comment.lower()  # Change to lowercase
    comment = re.sub("https?://\S+|www\.\S+", " ", comment)  # Remove links
    comment = re.sub("<.*?>+", " ", comment)  # Remove unwanted characters
    comment = re.sub("[%s]" % re.escape(string.punctuation), " ", comment)  # Remove punctuations
    comment = re.sub("\n", " ", comment)  # Remove next line symbols '\n'

    # Split the comment into individual words and only join the words that are not part of the
    # "STOPWORDS" set back into a single comment sentence.
    comment = " ".join(word for word in comment.split(" ") if word not in STOPWORDS)

    # Split the comment into individual words and only join the lemmatized words
    # back into a single comment sentence.
    clean_comment = " ".join(WORDNET_LEMMATIZER.lemmatize(word) for word in comment.split(" "))

    return clean_comment
