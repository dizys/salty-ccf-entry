import csv
import jieba
import jieba.analyse
import _pickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


STOP_WORD_LIST_PATH = '../../data/stop_words.txt'
TRAIN_DATA_PATH = '../../data/train.csv'
TEST_DATA_PATH = '../../data/test_public.csv'


def stopwordslist():
    stopwords = [line.strip()
                 for line in open(STOP_WORD_LIST_PATH, encoding='utf-8').readlines()]
    return stopwords


def subjectToId(subject):
    subjects = ['动力', '价格', '内饰', '配置', '安全性', '外观', '操控', '油耗', '空间', '舒适性']
    return subjects.index(subject)


def process(path):
    with open(path, mode='r', encoding='utf-8', newline='') as f:
        csv_file = csv.reader(f, dialect='excel')
        stopwords = stopwordslist()
        pre_row = ''
        result = []

        for (line, row) in enumerate(csv_file):
            if row[0] == 'content_id':
                continue

            is_new_id = (pre_row != row[0])

            if is_new_id:
                result.append([[], [], []])
            index = len(result) - 1

            if is_new_id:
                result[index][0] = segment(row[1], stopwords)
            result[index][1].append(subjectToId(row[2]))
            result[index][2].append(int(row[3]))
            pre_row = row[0]

        return result


def segment(text, stopwords):
    line_segments = []
    seg_list = jieba.cut(text, cut_all=False)

    for word in seg_list:
        word = word.strip()
        if len(word) > 0 and word not in stopwords:
            line_segments.append(word)

    return line_segments


def fetch_segments(data):
    segments = []

    for line in data:
        segments.append(line[0])

    return segments


def tfidf_vectorize(segments):
    vectorizer = TfidfVectorizer()
    corpus = []

    for (line, segment) in enumerate(segments):
        corpus.append(' '.join(segment))

    tfidf = vectorizer.fit_transform(corpus)

    return tfidf


def process_test_data(path):
    with open(path, mode='r', encoding='utf-8', newline='') as f:
        csv_file = csv.reader(f, dialect='excel')
        stopwords = stopwordslist()
        result = []

        for (line, row) in enumerate(csv_file):
            if row[0] == 'content_id':
                continue

            result.append([])
            index = len(result) - 1
            result[index] = segment(row[1], stopwords)

        return result


def main():
    data = process(TRAIN_DATA_PATH)
    print(data[0:100])

    segments = fetch_segments(data)
    tfidf = tfidf_vectorize(segments)
    print(tfidf)

    test_data = process_test_data(TEST_DATA_PATH)
    print(test_data[0:100])

    pickle.dump(data, open('../../data/pickles/data', 'wb'))
    pickle.dump(tfidf, open('../../data/pickles/data_tfidf', 'wb'))
    pickle.dump(test_data, open('../../data/pickles/test_data', 'wb'))


if __name__ == '__main__':
    main()
