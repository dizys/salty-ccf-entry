from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
          'This is the second second haha document.',
          'And the third one.',
          'Is this the first document?',
          ]


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

tfidf = tfidf.fit_transform(corpus)
print(tfidf[0])
print('------------')
print(tfidf[1])
print('------------')
print(tfidf[2])
print('------------')
print(tfidf[3])
print('------------')


print(type(tfidf))

print(tfidf.shape)
