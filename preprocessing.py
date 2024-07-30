import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np

def text_preprocessing(df):
    """
    This function does in-place replacement of data so it won't return anything
    """
    # Convert to lower cases
    df['Text'] = df['Text'].str.lower()

    # Remove punctuation
    df['Text'] = df['Text'].apply(lambda doc: re.sub(r'[^\w\s]+', '', doc))

    # Remove stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    df['Text'] = df['Text'].apply(lambda doc: ' '.join([word for word in doc.split() if word not in (stop_words)]))

    # Remove extra spaces
    df['Text'] = df['Text'].apply(lambda doc: re.sub(' +', ' ', doc))

    # Stemming
    porter_stemmer = PorterStemmer()
    df['Text'] = df['Text'].apply(lambda doc: [porter_stemmer.stem(word) for word in word_tokenize(doc)])
    df['Text'] = df['Text'].apply(lambda words: ' '.join(words))

