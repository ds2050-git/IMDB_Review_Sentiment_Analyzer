import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.callbacks import EarlyStopping



word_index=imdb.get_word_index()
import sys
sys.stdout.reconfigure(encoding='utf-8')

reverse_word_index={value: key for key, value in word_index.items()}


model=load_model('simple_rnn_imdb_review.h5')


def decode_review(sample_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in sample_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment,prediction[0][0]



positive_review="The movie was absolutely amazing with brilliant performances and a touching story."
sentiment,score=predict_sentiment(positive_review)
print("------------------------------------------------")
print(f"Review: {positive_review}")
print(f"Sentiment: {sentiment}")
print(f"Prediction Score: {score}")
print("------------------------------------------------")