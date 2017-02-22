from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
import numpy as np

folds =3

fvs = np.loadtxt("data/out_bag_of_words_5.csv", delimiter=',')
target = np.loadtxt("data/out_classes_5.txt")

## tfidf does nothing
# transformer = TfidfTransformer()
# fvs_tfidf = transformer.fit_transform(fvs)
# print(fvs_tfidf.shape)

print(type(fvs_tfidf))
clfmnb = MultinomialNB() #.fit(fvs, target)
scores = cross_val_score(clf, fvs_tfidf, target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clfbnb = BernoulliNB(binarize=0.5) #.fit(fvs, target)
scores = cross_val_score(clfbnb, fvs, target, cv=folds)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
