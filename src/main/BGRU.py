# !/usr/bin/python
# -*-coding:utf-8-*-
from keras.layers import Input, SpatialDropout1D, Dense
from keras.layers import Bidirectional, GRU, Flatten, Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.models import Model
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import jieba
import _pickle as pickle
import numpy as np
import keras.backend as K
K.clear_session()
gru_len = 128
Routings = 5
dropout_p = 0.25
rate_drop_dense = 0.28

# not enable in windows
# jieba.enable_parallel(4)

K.clear_session()

remove_stop_words = True

train_file = '../../data/train.csv'
test_file = '../../data/test_public.csv'

# load stopwords
with open('../../data/stop_words.txt', encoding='utf-8') as f:
    stop_words = set([l.strip() for l in f])

# load Glove Vectors
embeddings_index = {}
EMBEDDING_DIM = 300

# load data
train_df = pd.read_csv(train_file, encoding='utf-8')
test_df = pd.read_csv(test_file, encoding='utf-8')

train_df['label'] = train_df['subject'].str.cat(
    train_df['sentiment_value'].astype(str))

if remove_stop_words:
    train_df['content'] = train_df.content.map(
        lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
    test_df['content'] = test_df.content.map(
        lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
else:
    train_df['content'] = train_df.content.map(
        lambda x: ''.join(x.strip().split()))
    test_df['content'] = test_df.content.map(
        lambda x: ''.join(x.strip().split()))

train_dict = {}
for ind, row in train_df.iterrows():
    content, label = row['content'], row['label']
    if train_dict.get(content) is None:
        train_dict[content] = set([label])
    else:
        train_dict[content].add(label)

conts = []
labels = []
flag = -1
for k, v in train_dict.items():
    conts.append(k)
    labels.append(v)


def make_multilabel(labels):
    topic = ['动力', '价格', '内饰', '配置', '安全性',
             '外观', '操控', '油耗', '空间', '舒适性']
    sentiment = [-1, 0, 1]
    cmb = [str(i) + str(j) for i in topic for j in sentiment]
    index = [i for i in range(0, 30)]
    d = dict(zip(cmb, index))
    res = np.zeros((len(labels), 30), dtype=np.int32)
    i = 0
    for lst in labels:
        for item in lst:
            res[i][d[item]] = 1
        i += 1
    return [res.tolist()]


y_train = make_multilabel(labels)
# mlb = MultiLabelBinarizer()
# y_train = mlb.fit_transform(labels)
# with open('mlb.pickle', 'wb') as handle:
#     pickle.dump(mlb, handle)

content_list = [jieba.lcut(str(c)) for c in conts]

test_content_list = [jieba.lcut(c) for c in test_df.content.astype(str).values]

max_feature = 30000
tokenizer = text.Tokenizer(num_words=max_feature)
tokenizer.fit_on_texts(list(content_list) + list(test_content_list))

# saving tokenizer model
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)

seqs = tokenizer.texts_to_sequences(content_list)
seqs_dev = tokenizer.texts_to_sequences(test_content_list)

embedding_matrix = pickle.load(
    open('../../data/word2vec_model/embedding_matrix', 'rb'))


def get_padding_data(maxlen=100):
    x_train = sequence.pad_sequences(seqs, maxlen=maxlen)
    x_dev = sequence.pad_sequences(seqs_dev, maxlen=maxlen)
    return x_train, x_dev


def get_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(len(embedding_matrix),
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    x = Flatten()(x)
    x = Dropout(dropout_p)(x)
    output = Dense(30, activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


maxlen = 100
X_train, X_dev = get_padding_data(maxlen)
# print(X_train.shape, X_dev.shape, y_train.shape)


first_model_results = []
for i in range(5):
    model = get_model()
    model.fit(X_train, y_train, batch_size=64, epochs=15)  # 15
    first_model_results.append(model.predict(X_dev, batch_size=1024))
pred4 = np.average(first_model_results, axis=0)

model.save('model_7.h5')

tmp = [[i for i in row] for row in pred4]

for i, v in enumerate(tmp):
    if max(v) < 0.5:
        max_val = max(v)
        tmp[i] = [1 if j == max_val else 0 for j in v]
    else:
        tmp[i] = [int(round(j)) for j in v]

tmp = np.asanyarray(tmp)


def decode_multilabel(labels):
    topic = ['动力', '价格', '内饰', '配置', '安全性',
             '外观', '操控', '油耗', '空间', '舒适性']
    sentiment = [-1, 0, 1]
    cmb = [str(i) + str(j) for i in topic for j in sentiment]
    index = [i for i in range(0, 30)]
    d2 = dict(zip(index, cmb))
    x, y = labels.shape
    print(x, y)
    res = [[] for i in range(x)]
    for i in range(x):
        for j in range(y):
            if labels[i][j] == 1:
                res[i].append(d2[j])
    return res


# res = mlb.inverse_transform(tmp)
res = decode_multilabel(tmp)
cids = []
subjs = []
sent_vals = []
for c, r in zip(test_df.content_id, res):
    for t in r:
        if '-' in t:
            sent_val = -1
            subj = t[:-2]
        else:
            sent_val = int(t[-1])
            subj = t[:-1]
        cids.append(c)
        subjs.append(subj)
        sent_vals.append(sent_val)

res_df = pd.DataFrame({'content_id': cids, 'subject': subjs, 'sentiment_value': sent_vals,
                       'sentiment_word': ['便宜' for i in range(len(cids))]})

columns = ['content_id', 'subject', 'sentiment_value', 'sentiment_word']
res_df = res_df.reindex(columns=columns)
res_df.to_csv('submit_word.csv', encoding='utf-8', index=False)
