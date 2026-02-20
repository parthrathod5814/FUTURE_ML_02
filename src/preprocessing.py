import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download("stopwords")

nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"[^a-z\s]", "", text)

    doc = nlp(text)

    words = [
        token.lemma_
        for token in doc
        if token.text not in stop_words
        and len(token.text) > 2
    ]

    return " ".join(words)
