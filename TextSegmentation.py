import csv
import jieba
import jieba.analyse


def stopwordslist():
    stopwords = [line.strip() for line in open('stop_words.txt').readlines()]
    return stopwords


def segmentation():
    with open('train.csv', mode='r', encoding='utf-8', newline='') as f:
        # 读取csv文件
        csv_file = csv.reader(f, dialect='excel')
        stopwords = stopwordslist()
        pre_row = ''
        text_segmentation = ''
        for (line, row) in enumerate(csv_file):
            # 除重
            if row[1] != pre_row:
                # 分词
                seg_list = jieba.cut(row[1], cut_all=False)
                for word in seg_list:
                    if word not in stopwords:
                        text_segmentation += ' ' + word
                text_segmentation += '\n'
            pre_row = row[1]
        # print(text_segmentation)


if __name__ == '__main__':
    segmentation()
