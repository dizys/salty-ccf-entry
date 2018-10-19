# !/usr/bin/python
# -*-coding:utf-8-*-

import numpy as np

import _pickle as pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

texts = [['我', '时间', '爱', '天安门'],
         ['你', '爱', '北京', '爱'],
         ['引用', '全球', '威威', '规划', '但是', '环境', '离开'],
         ['引用', '全球', '时间']]

labels = [[1, 3], [2, 6], [4], [0, 5]]

MAX_NB_WORDS = 60
MAX_SEQUENCE_LENGTH = 5


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# labels = to_categorical(np.array(labels))
labels = MultiLabelBinarizer().fit_transform(labels)

pickle.dump(data, open('../../data/pickles/text_data', 'wb'))

