import jieba.posseg as jb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

##content_id,content,subject,sentiment_value,sentiment_word
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test_public.csv')
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
print(x_train)
x_test = tfidf[len(label):]
y_train = label

print(label)
numDimensions = 300
batchSize = 24
lstmUnits = 64
numClasses = 3
iterations = 50000
maxSeqLength = 500
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = x_train

#x_train,x_test,y_train,y_test=train_test_split(tfidf[0:len(label)],label,test_size=0.25,random_state=22)
print(x_train.shape)



sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(x_train, [1, 0, 2])
#取最终的结果值
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=label))
optimizer = tf.train.AdamOptimizer().minimize(loss)

for i in range(iterations):
    # Next Batch of reviews
    nextBatch, nextBatchLabels = x_train,y_train
    sess.run(optimizer, {x_train: nextBatch, label: nextBatchLabels})
    loss_ = sess.run(loss, {x_train: nextBatch, label: nextBatchLabels})
    accuracy_ = sess.run(accuracy, {x_train: nextBatch, label: nextBatchLabels})
    print("iteration {}/{}...".format(i + 1, iterations),
        "loss {}...".format(loss_),
        "accuracy {}...".format(accuracy_))
    # Save the network every 10,000 training iterations
    save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
    print("saved to %s" % save_path)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = x_test()
    print("Accuracy for this batch:", (sess.run(accuracy, {x_test: nextBatch, label: nextBatchLabels})) * 100)

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
