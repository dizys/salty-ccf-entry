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


def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    print(tfidf)
    return tfidf


def main():
    segments = segment()
    tfidf_vectorize(flatten(segments))


if __name__ == '__main__':
    main()
