from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import numpy as np

K = 5

# print("threshold: " + str(i))
# counts = np.loadtxt("data/counts.txt", delimiter=' ')
fvs = np.loadtxt("data/BOW.csv", delimiter=',')
print
target = np.loadtxt("data/out_classes_1.txt")
# print(words.shape, counts.shape)
# fvs_tfidf = np.column_stack([words,counts]) #, axis =1)
transformer = TfidfTransformer()
fvs_tfidf = transformer.fit_transform(fvs)

clfs = [MultinomialNB(), BernoulliNB(binarize=0.5), LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg'), LogisticRegression(), KNeighborsClassifier(n_neighbors=3), svm.LinearSVC(), SGDClassifier(), SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)]

for clf in clfs:
    print(clf)
    scores = cross_val_score(clf, fvs_tfidf, target, cv=K)
    # print(scores)
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print

thresholds = [1, 5]
for i in thresholds:
    print("threshold: " + str(i))
    counts = np.loadtxt("data/counts.txt", delimiter=' ')
    fvs = np.loadtxt("data/out_bag_of_words_" + str(i) + ".csv", delimiter=',')
    target = np.loadtxt("data/out_classes_" + str(i) + ".txt")
    # print(words.shape, counts.shape)
    # fvs_tfidf = np.column_stack([words,counts]) #, axis =1)
    transformer = TfidfTransformer()
    fvs_tfidf = transformer.fit_transform(fvs)
    

    for clf in clfs:
        print(clf)
        scores = cross_val_score(clf, fvs_tfidf, target, cv=K)
        # print(scores)
        print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print