import json
import random
import Sastrawi
import nltk
import warnings
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
lemmatizer = Lemmatizer()

model = tf.keras.models.load_model('InChatBot_Model.h5')

data = json.loads(open('indataset.json').read())

words = []
classes = []
documents = []
ignore_words = [".", ",", "?", "!", ")", "(", " "]

nltk.download('punkt')

#preprocessing
for intent in data['intents']:
  for pattern in intent['Pertanyaan']:
    #tokenisasi setiap kata
    w = nltk.word_tokenize(pattern)
    words.extend(w)

    #menambahkan dokumen ke korpus
    documents.append((w, intent['Tag']))

    #menambah ke our classes list
    if intent['Tag'] not in classes:
      classes.append(intent['Tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
#sort classes
classes = sorted(list(set(classes)))

def clean_up_sentence(sentence):
  #tokenisasi pertanyaan dengan mensplit kata ke array
  sentence_words = nltk.word_tokenize(sentence)
  #stem setiap kata - menjadikan kata dasar
  sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
  return sentence_words

def bow(sentence, words, show_details=True):
  #tokenisasi pertanyaan
  sentence_words = clean_up_sentence(sentence)
  #bag of words - matrix of n words, vocabulary matrix
  bag = [0]*len(words)
  for s in sentence_words:
    for i, w in enumerate(words):
      if w == s:
        #assign 1 if current word is in the vocabulary
        bag[i] = 1
        if show_details:
          print ("found in bag: %s" % w)
  return (np.array(bag))

def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))
    ERROR_THRESHOLD = 0.25
    # Ambil indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(res)
    # Ambil probabilitas kelas tertinggi
    predicted_probability = res[0][predicted_class_index]
    # Jika probabilitas di atas ambang batas, kembalikan kelas dan probabilitasnya
    if predicted_probability > ERROR_THRESHOLD:
        return [{"intent": classes[predicted_class_index], "probability": str(predicted_probability)}]
    else:
        return []


def getResponse(ints, data):
    tag = ints[0]['intent']
    list_of_intents = data['intents']
    for i in list_of_intents:
        if (i['Tag'] == tag):
            result = random.choice(i['Jawaban'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, data)
    return res


def get_bot_response():
  userText = request.args.get('msg')
  return chatbot_response(userText)

while True:
    message = input("")

    if message.lower() == 'keluar':
        print("Percakapan berakhir")
        break

    ints = predict_class(message)

    if ints:
      res = getResponse(ints, data)
      print(res)

    else:
      print("Maaf, saya tidak mengerti maksud anda")