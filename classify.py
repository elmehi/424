from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np

folds = 5

fvs = np.loadtxt("data/out_bag_of_words_5.csv", delimiter=',')
target = np.loadtxt("data/out_classes_5.txt")

## tfidf does nothing
# transformer = TfidfTransformer()
# fvs_tfidf = transformer.fit_transform(fvs)
# print(fvs_tfidf.shape)


clfs = [MultinomialNB(), BernoulliNB(binarize=0.5), LogisticRegression(), svm.LinearSVC()]
for clf in clfs:
    print(clf)
    scores = cross_val_score(clf, fvs, target, cv=folds)
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print