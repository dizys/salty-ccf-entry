# !/usr/bin/python
# -*-coding:utf-8-*-

import _pickle as pickle

a = pickle.load(open('../../data/pickles/text_data', 'rb'))

print(a.shape)

print(a)
