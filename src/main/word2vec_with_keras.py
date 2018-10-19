# !/usr/bin/python
# -*-coding:utf-8-*-

import sys
sys.path.append('../')

import gensim
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD

from utils.util import traditional_to_simplified

# file directory and name
WORD2VEC_DIR = '../../data/word2vec_model/'
WORD2VEC_NAME = 'zh.bin'
TEXT_DATA_DIR = '../../data/text/'


# IMPORTANT hyper_parameters
MAX_NB_WORDS = 99999
MAX_SEQUENCE_LENGTH = 99999
VALIDATION_SPLIT = 0.2


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

labels = to_categorical(np.asarray(labels))
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
3. make word2vec model
"""
model = gensim.models.Word2Vec.load(WORD2VEC_DIR + WORD2VEC_NAME)

word2idx = {"_PAD": 0} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
vocab_list = [[k, model.wv[k]] for k, v in model.wv.vocab.items()]

for i in range(len(vocab_list)):
    vocab_list[i][0] = traditional_to_simplified(vocab_list[i][0])

# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddings_matrix[i + 1] = vocab_list[i][1]


"""
4. make embedding layer
"""
EMBEDDING_DIM = 100 #词向量维度
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
          epochs=5, batch_size=128)


"""
7. test
"""
preds = model.predict(x_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
# score = compare preds and y_test
