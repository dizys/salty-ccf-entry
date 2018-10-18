import jieba.posseg as jb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

##content_id,content,subject,sentiment_value,sentiment_word
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test_public.csv')
print(train_csv)
corpus = []
'''
subject
0 价格
1 配置
2 操控
3 舒适性
4 油耗
5 动力
6 内饰
7 安全性
8 空间
9 外观
'''
from keras.utils import np_utils
hehe=list()
hehe.append('价格')
hehe.append('配置')
hehe.append('操控')
hehe.append('舒适性')
hehe.append('油耗')
hehe.append('动力')
hehe.append('内饰')
hehe.append('安全性')
hehe.append('空间')
hehe.append('外观')

label = list()
elabel = list()
for i in range(len(train_csv)):
    c = train_csv.content[i]

    s = ''
    for w in jb.cut(c):
        # print(w.word+'/'+w.flag)
        if 'a' in w.flag or 'l' in w.flag or 'v' in w.flag or 'zg' in w.flag or 'd' in w.flag or 'n' in w.flag:
            s = s + ' ' + w.word
    corpus.append(s)
    el = int(train_csv.sentiment_value[i]) + 1
    elabel.append(int(el))

    l = train_csv.subject[i]
    if l == '价格':
        label.append(0)
    if l == '配置':
        label.append(1)
    if l == '操控':
        label.append(2)
    if l == '舒适性':
        label.append(3)
    if l == '油耗':
        label.append(4)
    if l == '动力':
        label.append(5)
    if l == '内饰':
        label.append(6)
    if l == '安全性':
        label.append(7)
    if l == '空间':
        label.append(8)
    if l == '外观':
        label.append(9)

for i in range(len(test_csv)):
    c = test_csv.content[i]
    s = ''
    for w in jb.cut(c):
        # print(w.word+'/'+w.flag)
        if 'a' in w.flag or 'l' in w.flag or 'v' in w.flag or 'zg' in w.flag or 'd' in w.flag or 'n' in w.flag:
            s = s + ' ' + w.word
    corpus.append(s)

# label = np_utils.to_categorical(label)
#
# cv = CountVectorizer()
#
# tfidft =TfidfTransformer()
# bow = cv.fit_transform(corpus)
stop_words = list()
with open("stopwords.txt",'r') as fff:
    for line in fff.readlines():
        linestr=line.strip()
        stop_words.append(linestr)


vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop_words, ngram_range=(1, 2), norm='l2')
tfidf = vectorizer.fit_transform(corpus)
x_train = tfidf[0:len(label)]
x_test = tfidf[len(label):]
y_train = label
# x_train,x_test,y_train,y_test=train_test_split(tfidf[0:len(label)],label,test_size=0.25,random_state=22)
print(x_train.shape)

log_reg = LogisticRegression(class_weight="balanced")
log_reg.fit(x_train, y_train)
emo_reg = LogisticRegression(class_weight="balanced")
emo_reg.fit(x_train, elabel)

acc = 0
predictions = log_reg.predict_proba(x_test)
with open('car.csv', mode='w',encoding='utf-8') as f:
    f.write("content_id,subject,sentiment_value,sentiment_word"+'\n')
    for i, prediction in enumerate(predictions[:]):
        # print ('Prediction:%s. ' % (np.where(prediction>0.10)))
        ers = emo_reg.predict(x_test[i]) - 1
        loc = np.array(np.where(prediction > 0.10))
        for pos in range(len(loc[0])):
            f.write(test_csv.content_id[i] + ',' + hehe[loc[0][pos]] + ',' + str(ers[0]) + ',' + '\n')
