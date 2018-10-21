# !/usr/bin/python
# -*-coding:utf-8-*-

import _pickle as pickle

import sys
sys.path.append('../')

import gensim
import numpy as np

# from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Input, Dense, Dropout, Activation, Embedding

from sklearn.preprocessing import MultiLabelBinarizer
from utils.util import traditional_to_simplified

# file directory and name
WORD2VEC_DIR = '../../data/word2vec_model/'
WORD2VEC_NAME = 'zh.bin'
TEXT_DATA_DIR = '../../data/text/'


# IMPORTANT hyper_parameters
MAX_NB_WORDS = 15000
MAX_SEQUENCE_LENGTH = 40
VALIDATION_SPLIT = 0.9


data = pickle.load(open('../../data/pickles/data', 'rb'))

# print(data.shape)

print(len(data))
print(len(data[0]))

from pprint import pprint

print(data[:2])
