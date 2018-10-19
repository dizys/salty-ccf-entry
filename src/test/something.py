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

print(embeddings_matrix.shape)
