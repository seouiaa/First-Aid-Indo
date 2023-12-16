import json
import random
import Sastrawi
import nltk
import warnings
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from keras.models import load_model
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
lemmatizer = Lemmatizer()

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

#documents = kombinasi antara patterns dan intents
print (len(documents), 'documents')

#classes = intents
print (len(classes), 'classes', classes)

#words = semua kata, vocabulary
print (len(words), 'unique lemmatized words', words)

#training
training = []

#membuat array kosong untuk output
output_empty = [0] * len(classes)

#training set, bag of words untuk setiap kalimat
for doc in documents:
  #inisialisasi bag of words
  bag = []
  #daftar tokenisasi kata untuk pertanyaan
  pattern_words = doc[0]

  #lematisasi setiap kata, membuat kata dasar, untuk representasi kata yang sesuai
  pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

  #membuat bag of words array dengan 1, jika kata cocok ditemukan di pola terkini
  for w in words:
    bag.append(1) if w in pattern_words else bag.append(0)
  #output berupa '0' untuk seriap tag dan '1' untuk tag terbaru (untuk setiap pertanyaan)
  output_row = list(output_empty)
  output_row[classes.index(doc[1])] = 1
  training.append([bag, output_row])

#acak fitur dan konversi ke numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

#membuat train dan test list
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")

#modeling
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#compile model
sgd = SGD(learning_rate=0.001, momentum = 0.9, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting dan menyimpan model
history = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=2)
model.save('InChatBot_Model.h5', history)
accuracy = history.history['accuracy'][-1] * 100
print('\n')
print('*'*100)
print("\nPembuatan model sukses")
print(f"Akurasi sebesar: {accuracy:.2f}%")