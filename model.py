from flask import Flask, request
import numpy as np 
import pandas as pd
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

model = tf.keras.models.load_model('new_model.h5')

data = pd.read_csv('Macam Infeksi Gigi.xlsx - DataTwit.csv', usecols=['Kategori', 'Gejala'])
data.dropna(inplace=True)
data.head()

num_of_categories = 45000
shuffled = data.reindex()
a = shuffled[shuffled['Kategori'] == 'a'][:num_of_categories]
b = shuffled[shuffled['Kategori'] == 'b'][:num_of_categories]
c = shuffled[shuffled['Kategori'] == 'c'][:num_of_categories]
d = shuffled[shuffled['Kategori'] == 'd'][:num_of_categories]
e = shuffled[shuffled['Kategori'] == 'e'][:num_of_categories]
f = shuffled[shuffled['Kategori'] == 'f'][:num_of_categories]

concated = pd.concat([a,b,c,d,e,f], ignore_index=True)
concated['LABEL'] = 0

concated.loc[concated['Kategori'] == 'a', 'LABEL'] = 0
concated.loc[concated['Kategori'] == 'b', 'LABEL'] = 1
concated.loc[concated['Kategori'] == 'c', 'LABEL'] = 2
concated.loc[concated['Kategori'] == 'd', 'LABEL'] = 3
concated.loc[concated['Kategori'] == 'e', 'LABEL'] = 4
concated.loc[concated['Kategori'] == 'f', 'LABEL'] = 5
labels = to_categorical(concated['LABEL'], num_classes=6)
if 'Kategori' in concated.keys():
    concated.drop(['Kategori'], axis=1)
'''
[1. 0. 0. 0. 0. 0.] a
[0. 1. 0. 0. 0. 0.] b
[0. 0. 1. 0. 0. 0.] c
[0. 0. 0. 1. 0. 0.] d
[0. 0. 0. 0. 1. 0.] e
[0. 0. 0. 0. 0. 1.] f
'''

n_most_common_words = 8000
max_len = 130
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(concated['Gejala'].values)
sequences = tokenizer.texts_to_sequences(concated['Gejala'].values)
word_index = tokenizer.word_index

app = Flask(__name__)
@app.route('/')
def index():
    return "HELLO WORLD"

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    input = []
    input.append(text)
    seq = tokenizer.texts_to_sequences(input)
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    labels = ['Kemungkinan kamu mengalami karies gigi. Jangan lupa untuk mengunjungi dokter gigi terdekat dan lakukan perawatan saluran akar gigi ya', 'Kemungkinan kamu mengalamin karang Gigi. Jangan lupa untuk mengunjungi dokter gigi terdekat dan lakukan scalling gigi ya', 'Kemungkinan kamu mengalami abses gigi. Jangan lupa untuk mengunjungi dokter gigi terdekat dan membeli obat antibiotik atau cabut gigi ya', 'Kemungkinan kamu mengalami radang gusi. Jangan lupa untuk mengunjungi dokter gigi terdekat dan lakukan pembersihan karang gigi (scaling) atau penambalan gigi ya','Maaf kami tidak bisa memprediksi gejala penyakit yang kamu alami, namun jika mengganggu aktivitas sehari hari silahkan mengunjungi dokter gigi terdekat ya','Kemungkinan kamu mengalami sariawan. Jangan lupa untuk tetesi luka sariawan dengan obat yang mengandung antibiotik ya']
    return labels[np.argmax(pred)]

if __name__ == '__main__':
    app.run()