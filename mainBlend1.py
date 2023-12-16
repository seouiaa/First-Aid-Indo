import json
import random
import nltk
import numpy as np
import tensorflow as tf
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()

user_input = int(input("Silahkan pilih bahasa yang diinginkan (1: Indonesia, 2: English): "))

if user_input == 1:
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
            wrds = nltk.word_tokenize(pattern)
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

elif user_input == 2:
    model = tf.keras.models.load_model('EngChatbotModel.h5')

    with open('EnglishData.json', 'r') as file:
        data = json.load(file)

    # Extract words and labels
    words = []
    labels = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # Word Stemming
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # Initialize NLTK stemmer
    stemmer = LancasterStemmer()

    # Function to convert input to a bag of words
    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return [bag]

    # Function to get a response from the chatbot
    def get_response(inp):
        results = model.predict(np.array(bag_of_words(inp, words)))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for intent in data['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']

        return random.choice(responses)

    # Main chat loop
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        response = get_response(inp)
        print(response)
