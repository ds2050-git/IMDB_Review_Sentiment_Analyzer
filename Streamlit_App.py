import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}


model=load_model('simple_rnn_imdb_review.h5')


def decode_review(sample_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in sample_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[]
    for word in words:
        index=word_index.get(word)
        if index is not None:
            index+=2
            if index<10000:
                encoded_review.append(index)
            else:   
                encoded_review.append(2)
        else:
            encoded_review.append(2)

    if not encoded_review:
        encoded_review=[2]
        
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input=st.text_area('Movie Review')

if st.button('Classify'):
    if len(user_input.strip())==0:
        st.warning("Please enter a valid review.")
    else:
        preprocess_input=preprocess_text(user_input)
        prediction=model.predict(preprocess_input)
        sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'


        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')