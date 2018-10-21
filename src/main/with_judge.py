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
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Input, Dense, Dropout, Activation, Embedding

from utils.util import traditional_to_simplified
from sklearn.linear_model import LogisticRegression

# file directory and name
WORD2VEC_DIR = '../../data/word2vec_model/'
WORD2VEC_NAME = 'embedding_matrix'
DATA_DIR = '../../data/pickles/'
TRAIN_DATA_NAME = 'data'
TEST_DATA_NAME = 'test_data'

# IMPORTANT hyper_parameters
MAX_NB_WORDS = 15000
MAX_SEQUENCE_LENGTH = 25
VALIDATION_SPLIT = 0.2

EMBEDDING_DIM = 300
JUDGE_THRESHOLD = 0.2
MAX_EPOCHES = 50
BATCH_SIZE = 32

sub_map = ['动力', '价格', '内饰', '配置', '安全性', '外观', '操控', '油耗', '空间', '舒适性']
sen_map = ['-1', '0', '1']
"""
1. get texts
"""
# waiting for cleaned data.
train_data = pickle.load(open(DATA_DIR + TRAIN_DATA_NAME, 'rb'))
print(train_data[:2])

train_texts = [x[0] for x in train_data]
label_sub = [x[1][0] for x in train_data]
label_sen = [x[2][0]+1 for x in train_data]
# print(label_sen[:30])
# print(len(sentiment[0]))

test_data = pickle.load(open(DATA_DIR + TEST_DATA_NAME, 'rb'))
test_id = [x[0] for x in test_data]
test_texts = [x[1] for x in test_data]

"""
2. pre_processing
"""
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

train_texts = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_texts = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

label_sub = to_categorical(label_sub)
label_sen = to_categorical(label_sen)  # TODO

"""
3. get train, test, validate data
"""

indices = np.arange(train_texts.shape[0])
np.random.shuffle(indices)
train_texts = train_texts[indices]
label_sub = label_sub[indices]
nb_validation_samples = int(VALIDATION_SPLIT * train_texts.shape[0])
spt = nb_validation_samples // 2

x_train = train_texts[:-nb_validation_samples]
sub_train = label_sub[:-nb_validation_samples]
sen_train = label_sen[:-nb_validation_samples]

x_val = train_texts[-nb_validation_samples:-spt]
sub_val = label_sub[-nb_validation_samples:-spt]
sen_val = label_sen[-nb_validation_samples:-spt]

x_judge = train_texts[-spt:]
sub_judge = label_sub[-spt:]
sen_judge = label_sen[-spt:]

x_test = test_texts

#
# print(x_test[0])
# print('---------')
# print(x_test[1])
# print('---------')
# print(x_test[2])
# print('---------')
"""
4. load word2vec model
"""
embeddings_matrix = pickle.load(open(WORD2VEC_DIR + WORD2VEC_NAME, 'rb'))
# print(embeddings_matrix.shape)
# (50102, 300)


"""
5. make embedding layer
"""
embedding_layer = Embedding(len(embeddings_matrix),
                            EMBEDDING_DIM,
                            weights=[embeddings_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

input_x = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
x = embedding_layer(input_x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Flatten()(x)
output_1 = Dense(10, activation='softmax')(x)

input_y = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
y = embedding_layer(input_y)
# y = Dense(128, activation='relu')(y)
# y = Dropout(0.5)(y)
y = Dense(64, activation='relu')(y)
y = Flatten()(y)
output_2 = Dense(3, activation='softmax')(y)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model_1 = Model(input_x, output_1)
model_1.compile(loss='binary_crossentropy',
              optimizer=sgd,  # sgd   RMSprop()
              metrics=['acc'])

model_2 = Model(input_y, output_2)
model_2.compile(loss='binary_crossentropy',  # 'categorical_crossentropy',
              optimizer=sgd,  # sgd   RMSprop()
              metrics=['acc'])


"""
6. learn
"""
model_1.fit(x_train, sub_train,
          validation_data=(x_val, sub_val),
          epochs=MAX_EPOCHES, batch_size=BATCH_SIZE,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

model_2.fit(x_train, sen_train,
          validation_data=(x_val, sen_val),
          epochs=MAX_EPOCHES, batch_size=BATCH_SIZE,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])


"""
7. test
"""
preds = model_1.predict(x_judge)
ans = [x.argmax() for x in preds]
from collections import Counter
c = Counter(ans)
print(c)
print('------------------------')
sentiment_preds = model_2.predict(x_judge)
ans2 = [x.argmax() for x in sentiment_preds]
c2 = Counter(ans2)
print(c2)
print('------------------------')

num_1 = 0
num_2 = 0
lenss = len(ans)

print(ans[0])
print(sub_judge[0])

for x in range(len(ans)):
    if sub_judge[x].argmax() == ans[x]:
        num_1 += 1
    if sen_judge[x].argmax() == ans2[x]:
        num_2 += 1

print('acc_sub: %.6f, acc_sen: %.6f.' % (num_1 / lenss, num_2 / lenss))


# with open('r5.csv', mode='w', encoding='utf-8') as f:
#     f.write("content_id,subject,sentiment_value,sentiment_word"+'\n')
#     for i in range(len(ans)):
#         f.write(test_id[i] + ',' + sub_map[ans[i]] + ',' + str(sen_map[ans2[i]]) + ',' + '\n')
#
