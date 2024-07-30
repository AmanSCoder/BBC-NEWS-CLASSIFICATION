import gradio as gr
import joblib
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


def predict_user_input(paragraph, tfidf, nmf, label_mapping_yp):
    data = pd.DataFrame({'Text': [paragraph]})
    text_preprocessing(data)
    tfidf_transformed = tfidf.transform(data['Text'])
    nmf_transformed = nmf.transform(tfidf_transformed)
    y_pred = np.argmax(nmf_transformed, axis=1)
    y_pred = [label_mapping_yp[y] for y in y_pred]
    return y_pred[0]

def process_paragraph(paragraph):
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    nmf = joblib.load('nmf_model.pkl')
    label_mapping_yp = joblib.load('label_mapping.pkl')
    predicted_class = predict_user_input(paragraph, tfidf, nmf, label_mapping_yp)
    print(f"The predicted class for the input paragraph is: {predicted_class}")
    return predicted_class

def paragraph_processing_app(paragraph):
    processed_text = process_paragraph(paragraph)
    return processed_text

input_text = gr.Textbox(lines=10, label="Enter a article:")
output_text = gr.Textbox(label="Category(Out of Business, Tech, Sport, Politics and Entertainment.)")

gr.Interface(fn=paragraph_processing_app, inputs=input_text, outputs=output_text).launch()
