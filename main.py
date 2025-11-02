import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import pickle
import tensorflow as tf

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, ' ') for i in encoded_review])


def preprocess_text(text):
    text = str(text)
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# streamlit app


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area("Movie Review")

if st.button('Classify'):
    # Debug: Show preprocessed text
    preprocessed_input = preprocess_text(user_input)
    
    # Get raw prediction
    prediction = model.predict(preprocessed_input, verbose=0)
    raw_score = float(prediction[0][0])
    
    # Apply threshold with stronger criteria
    if raw_score > 0.6:  # Increased threshold for positive sentiment
        sentiment = 'Positive'
    elif raw_score < 0.4:  # Clear negative sentiment
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'  # For scores between 0.4 and 0.6
    
    # Show results with more detail
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Raw Prediction Score: {raw_score:.4f}')
    
    # Add debug information
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Input text length: {len(user_input.split())}")
    st.sidebar.write(f"Input words found in vocabulary: {sum(1 for w in user_input.lower().split() if w in reverse_word_index.values())}")
else:
    st.write("Please enter a movie review")
