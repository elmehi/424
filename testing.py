from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn import svm
import numpy as np


fvs_train = np.loadtxt("data/BOW.csv", delimiter=',')
target_train = np.loadtxt("data/out_classes_1.txt")
fvs_test = np.loadtxt("data/bow_test.csv", delimiter=',')
target_test = np.loadtxt("data/test_classes.txt")

transformer = TfidfTransformer()
fvs_train = transformer.fit_transform(fvs_train)
fvs_test = transformer.transform(fvs_test)

c = MultinomialNB()
c.fit(fvs_train, target_train)
predicted = c.predict(fvs_test)
print("mean:", np.mean(predicted == target_test))
print(metrics.classification_report(target_test, predicted))
