import csv
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORD_LIST_PATH = 'data/stop_words.txt'
TRAIN_DATA_PATH = 'data/train.csv'


def flatten(list):
    return [item for sublist in list for item in sublist]


def stopwordslist():
    stopwords = [line.strip()
                 for line in open(STOP_WORD_LIST_PATH).readlines()]
    return stopwords


def segment():
    with open(TRAIN_DATA_PATH, mode='r', encoding='utf-8', newline='') as f:
        # 读取csv文件
        csv_file = csv.reader(f, dialect='excel')
        stopwords = stopwordslist()
        pre_row = ''
        text_segmentation = []
        for (line, row) in enumerate(csv_file):
            # 除重
            if row[1] != pre_row:
                line_segmentation = []
                # 分词
                seg_list = jieba.cut(row[1], cut_all=False)
                for word in seg_list:
                    if word not in stopwords:
                        line_segmentation.append(word)
                text_segmentation.append(line_segmentation)
            pre_row = row[1]
        return text_segmentation


def sub():
    with open('train.csv', mode='r', encoding='utf-8', newline='') as f:
        # 读取csv文件
        csv_file = csv.reader(f, dialect='excel')
        pre_row = ''
        tmp_line = 1
        subject = []
        for (line, row) in enumerate(csv_file):
            subject.append([])
            # 除重
            if row[1] != pre_row:
                # 记录去重后初次出现的行数
                tmp_line = line
                subject[line].append(row[2])
            else:
                # 如果与上一条评论相同，则将主题和评分放入初次出现行中的list
                subject[tmp_line].append(row[2])
            pre_row = row[1]

        return subject


def value():
    with open('train.csv', mode='r', encoding='utf-8', newline='') as f:
        # 读取csv文件
        csv_file = csv.reader(f, dialect='excel')
        pre_row = ''
        tmp_line = 1
        sentiment_value = []
        for (line, row) in enumerate(csv_file):
            sentiment_value.append([])
            # 除重
            if row[1] != pre_row:
                # 记录去重后初次出现的行数
                tmp_line = line
                sentiment_value[line].append(row[3])
            else:
                # 如果与上一条评论相同，则将主题和评分放入初次出现行中的list
                sentiment_value[tmp_line].append(row[3])
            pre_row = row[1]
        return sentiment_value


def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    print(tfidf)
    return tfidf


def main():
    segments = segment()
    subject = sub()
    setiment_value = value()
    tfidf_vectorize(flatten(segments))


if __name__ == '__main__':
    main()
