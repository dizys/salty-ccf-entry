# !/usr/bin/python
# -*-coding:utf-8-*-

import sys
sys.path.append('../')

import gensim
import numpy as np
import _pickle as pickle
from sklearn.preprocessing import MultiLabelBinarizer

from keras.models import Model
from keras.optimizers import SGD
# from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Input, Dense, Dropout, Activation, Embedding

from utils.util import traditional_to_simplified

# file directory and name
WORD2VEC_DIR = '../../data/word2vec_model/'
WORD2VEC_NAME = 'embedding_matrix'
TEXT_DATA_DIR = '../../data/text/'


# IMPORTANT hyper_parameters
MAX_NB_WORDS = 15000
MAX_SEQUENCE_LENGTH = 40
VALIDATION_SPLIT = 0.9

EMBEDDING_DIM = 300
JUDGE_THRESHOLD = 0.5
MAX_EPOCHES = 2
BATCH_SIZE = 32


"""
1. get texts
"""
# waiting for cleaned data.
texts = []
labels = []


"""
2. preprocessing and get train, test, validate data
"""
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# labels = to_categorical(np.asarray(labels))
labels = MultiLabelBinarizer().fit_transform(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


"""
3. load word2vec model
"""
embeddings_matrix = pickle.load(open(WORD2VEC_DIR + WORD2VEC_NAME))
# print(embeddings_matrix.shape)
# (50102, 300)


"""
4. make embedding layer
"""
embedding_layer = Embedding(len(embeddings_matrix),
                            EMBEDDING_DIM,
                            weights=[embeddings_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


"""
5. make model
"""
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)

# for multilabel problem
preds = Dense(len(labels_index), activation='sigmod')(x)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',  # sgd
              metrics=['acc'])


"""
6. learn
"""
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=MAX_EPOCHES, batch_size=BATCH_SIZE)


"""
7. test
"""
preds = model.predict(x_test)
preds[preds >= JUDGE_THRESHOLD] = 1
preds[preds < JUDGE_THRESHOLD] = 0
# score = compare preds and y_test
