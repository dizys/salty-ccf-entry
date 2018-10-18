import csv
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer


def stopwordslist():
    stopwords = [line.strip() for line in open('stop_words.txt').readlines()]
    return stopwords


def segmentation():
    with open('train.csv', mode='r', encoding='utf-8', newline='') as f:
        # 读取csv文件
        csv_file = csv.reader(f, dialect='excel')
        stopwords = stopwordslist()
        pre_row = ''
        tmp_line = 1
        text_segmentation = []
        subject = []
        sentiment_value = []
        for (line, row) in enumerate(csv_file):
            text_segmentation.append([])
            subject.append([])
            sentiment_value.append([])
            # 除重
            if row[1] != pre_row:
                # 分词
                seg_list = jieba.cut(row[1], cut_all=False)
                for word in seg_list:
                    if word not in stopwords:
                        text_segmentation[line].append(word)
                # 记录去重后初次出现的行数
                tmp_line = line
                subject[line].append(row[2])
                sentiment_value[line].append(row[3])
            else:
                # 如果与上一条评论相同，则将主题和评分放入初次出现行中的list
                subject[tmp_line].append(row[2])
                sentiment_value[tmp_line].append(row[3])
            # print(row[1] + ' empty segment')
            # print(seg_list)
            pre_row = row[1]
        print(subject)
        print(sentiment_value)
        print(text_segmentation)


if __name__ == '__main__':
    segmentation()
