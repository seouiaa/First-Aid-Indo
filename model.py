import json
import random
import Sastrawi
import nltk
import warnings
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from keras.models import load_model
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
lemmatizer = Lemmatizer()

data = json.loads(open('indataset.json').read())

# Extracting Data

words = []
labels = []
docs_x = []
docs_y = []
ignore_words = [".", ",", "?", "!", ")", "(", " "]

for intent in data['intents']:
    for pattern in intent['Pertanyaan']:
        wrds=nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['Tag'])

    if intent['Tag'] not in labels:
        labels.append(intent['Tag'])

# Word Steamming
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

labels = sorted(labels)

# Bag of Words
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Split Data
train_data, val_data, train_output, val_output = train_test_split(training, output, test_size=0.25, random_state=42)

# Developing a Model
model = Sequential([
    Dense(32, input_shape=(len(training[0]),), activation='relu'),
    Dropout(0.4),
    Dense(16, activation='relu'),
    Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(train_data, train_output, epochs=500, batch_size=32, validation_data=(val_data, val_output))
model.save('InChatBot_Model.h5', history)
accuracy = history.history['accuracy'][-1] * 100
print('\n')
print('*'*100)
print("\nPembuatan model sukses")
print(f"Akurasi sebesar: {accuracy:.2f}%")