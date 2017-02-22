from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
import numpy as np

fvs = np.loadtxt("data/out_bag_of_words_5.csv", delimiter=',')
target = np.loadtxt("data/out_classes_5.txt")
print(fvs[1:10])

## tfidf does nothing
transformer = TfidfTransformer()
fvs_tfidf = transformer.fit_transform(fvs)
# print(fvs_tfidf.shape)
print(fvs_tfidf[1:10])
clf = MultinomialNB() #.fit(fvs, target)


scores = cross_val_score(clf, fvs_tfidf, target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))