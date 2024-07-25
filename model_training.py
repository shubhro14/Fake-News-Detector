
import pandas as pd
import numpy as np
import tensorflow as tf


import re
import nltk
import warnings

import pickle

warnings.filterwarnings('ignore')

df1=pd.read_csv('/Dataset/train.csv')

df2=pd.read_csv("/Dataset/news.csv")


df1=df1.drop(columns=['id','title','author'],axis=1)
df1 = df1.dropna(axis=0)


df2=df2.drop(columns=['Unnamed: 0','title'],axis=1)
df2 = df2.dropna(axis=0)
label_mapping = {'REAL': 0, 'FAKE': 1}
df2['label'] = df2['label'].map(label_mapping)


df = pd.concat([df1, df2], ignore_index=True)


# remove special characters and punctuations
df['news'] = df['text'].str.lower()
df['news'] = df['news'].str.replace('[^A-Za-z0-9\s]', '')
df['news'] = df['news'].str.replace('\n', '')
df['news'] = df['news'].str.replace('\s+', ' ')


# remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
df['news'] = df['news'].apply(lambda x: " ".join([word for word in x.split() if word not in stop]))


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['news'])
word_index = tokenizer.word_index
vocab_size = len(word_index)

sequences = tokenizer.texts_to_sequences(df['news'])
padded_seq = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
with open('/Fake_News/tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
# embedding index
embedding_index = {}
with open('/content/drive/MyDrive/Fake news detection _dataset/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
# embedding matrix
embedding_matrix = np.zeros((vocab_size+1, 100))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(padded_seq, df['label'], test_size=0.20, random_state=42, stratify=df['label'])

from keras.layers import LSTM, Dropout, Dense, Embedding
from keras import Sequential

model = Sequential([
    Embedding(vocab_size+1, 100, weights=[embedding_matrix], trainable=False),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(256),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

history = model.fit(x_train, y_train, epochs=25, batch_size=256, validation_data=(x_test, y_test))


model.save('/Fake_News/model.h5')
