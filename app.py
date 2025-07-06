import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.callbacks import EarlyStopping

max_features=10000
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)



sample_review=x_train[0]
sample_label=y_train[0]


word_index=imdb.get_word_index()
import sys
sys.stdout.reconfigure(encoding='utf-8')

reverse_word_index={value: key for key, value in word_index.items()}


decoded_review=' '.join([reverse_word_index.get(i-3,'?') for i in sample_review])


max_len=500
x_train=sequence.pad_sequences(x_train,maxlen=max_len) 
x_test=sequence.pad_sequences(x_test,maxlen=max_len)


model=Sequential()
feature_dimensions=128 # Using 128 features in embedding layer
neurons=128
model.add(Embedding(max_features,feature_dimensions)) # Embedding Layer
model.add(SimpleRNN(neurons,activation='relu')) # Simple RNN layer
model.add(Dense(1,activation="sigmoid")) # ONly 1 output hideen layer

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

history=model.fit(x_train,y_train,epochs=5,batch_size=64,validation_split=0.2,callbacks=[earlystopping])

model.save('simple_rnn_imdb_review.h5')

