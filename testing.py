from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn import svm
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

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
print(fvs_test.shape)
p_score = c.predict_proba(fvs_test)
# print(p_score)
# print(metrics.roc_curve(target_test,p_score[:,0]))
targets2d = np.array([1-target_test,target_test])
print(targets2d)
targets2d = np.transpose(targets2d)
print(targets2d)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
# print(p_score)
# print(targets2d)

for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(targets2d[:, i], p_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(targets2d.ravel(), p_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



plt.figure()
lw = 2
cl = 1
plt.plot(fpr[cl], tpr[cl], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[cl])

cl = 0
plt.plot(fpr[cl], tpr[cl], color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[cl])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
