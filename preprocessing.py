import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

import spacy
import logging
# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    logging.debug("Original text: %s", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    processed_text = ' '.join(tokens)
    logging.debug("Processed text: %s", processed_text)
    return processed_text

def load_and_preprocess_data(file_path):
    yelp_data = pd.read_csv(file_path)
    yelp_data['cleaned_text'] = yelp_data['text'].apply(preprocess_text)
    yelp_data['sentiment'] = yelp_data['stars'].apply(assign_sentiment)
    return yelp_data

def assign_sentiment(stars):
    if stars in [1, 2]:
        return 'negative'
    elif stars == 3:
        return 'neutral'
    else:
        return 'positive'
