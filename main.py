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
model = tf.keras.models.load_model('InChatBot_Model.h5')

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

# Making Predictions
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array([bag])

def get_response(inp, data, threshold=0.5):
    results = model.predict(bag_of_words(inp, words))[0]
    max_prob = np.max(results)

    if max_prob < threshold:
        return "Maaf, saya tidak mengerti maksud Anda."

    results_index = np.argmax(results)
    tag = labels[results_index]

    matching_intent = next((intent for intent in data["intents"] if intent['Tag'] == tag), None)

    if matching_intent:
        return random.choice(matching_intent['Jawaban'])
    else:
        return "Maaf, saya tidak mengerti maksud Anda."

def chat():
    print("Mulai percakapan dengan bot, ketik 'keluar' untuk mengakhiri percakapan")
    while True:
        inp = input("Anda: ")
        if inp.lower() == "keluar":
            print("Percakapan berakhir")
            break

        response = get_response(inp, data)
        print(response)

chat()