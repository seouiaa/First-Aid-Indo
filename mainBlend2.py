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
    import Imain
    Imain

elif user_input == 2:
    import Emain
    Emain