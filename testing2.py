from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier

thresholds = [1,5]
for i in thresholds:
    print 'treshold =', i
    fvs_train = np.loadtxt("data/out_bag_of_words_" + str(i) + ".csv", delimiter=',')
    target_train = np.loadtxt("data/out_classes_1.txt")
    fvs_test = np.loadtxt("data/testout_"+str(i)+"_bag_of_words_" + str(i) + ".csv", delimiter=',')
    target_test = np.loadtxt("data/test_classes.txt")

    transformer = TfidfTransformer()
    fvs_train = transformer.fit_transform(fvs_train)
    fvs_test = transformer.transform(fvs_test)

    classes = [('mn', MultinomialNB()), ('bn', BernoulliNB(binarize=0.5)), ('lr', LogisticRegression()), ('sgd', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for c in classes: 
        clf = c[1].fit(fvs_train, target_train)
        predicted = clf.predict(fvs_test)
        print(c[0], ":")
        print("mean:", np.mean(predicted == target_test))
        print(metrics.classification_report(target_test, predicted))
        print(fvs_test.shape)
        if c[0] is not 'sgd':
            p_score = clf.predict_proba(fvs_test)
            # print(metrics.roc_curve(target_test,p_score[:,0]))
            targets2d = np.array([1-target_test,target_test])
            targets2d = np.transpose(targets2d)

            # Compute ROC curve and ROC area for each class
            print(c[0])
            fpr[c[0]], tpr[c[0]], _ = roc_curve(targets2d[:, 0], p_score[:, 0])
            roc_auc[c[0]] = auc(fpr[c[0]], tpr[c[0]])

    # plt.figure()
    # print(fpr.keys())
    # lw = 2
    # cl = 'mn'
    # plt.plot(fpr[cl], tpr[cl], color='darkorange',
    #          lw=lw, label='Multinomial Naive Bayes (area = %0.2f)' % roc_auc[cl])

    # cl = 'bn'
    # plt.plot(fpr[cl], tpr[cl], color='red',
    #          lw=lw, label='Bernoulli Naive Bayes (area = %0.2f)' % roc_auc[cl])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')


    # cl = 'lr'
    # plt.plot(fpr[cl], tpr[cl], color='black',
    #          lw=lw, label='Logistic Regression (area = %0.2f)' % roc_auc[cl])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC (threshold = ' + str(i) + ')')
    # plt.legend(loc="lower right")
    # plt.savefig("LaTeX/ROC.png")
    # plt.show()


print("w. feature selection")

fvs_train = np.loadtxt("data/BOW.csv", delimiter=',')
target_train = np.loadtxt("data/out_classes_1.txt")
fvs_test = np.loadtxt("data/bow_test.csv", delimiter=',')
target_test = np.loadtxt("data/test_classes.txt")

transformer = TfidfTransformer()
fvs_train = transformer.fit_transform(fvs_train)
fvs_test = transformer.transform(fvs_test)

classes = [('mn', MultinomialNB()), ('bn', BernoulliNB(binarize=0.5)), ('lr', LogisticRegression()), ('sgd', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]

fpr = dict()
tpr = dict()
# microfpr = dict()
# microtpr = dict()
roc_auc = dict()

for c in classes: 
    clf = c[1].fit(fvs_train, target_train)
    predicted = clf.predict(fvs_test)
    print(c[0], ":")
    print("mean:", np.mean(predicted == target_test))
    print(metrics.classification_report(target_test, predicted))
    print(fvs_test.shape)
    if c[0] is not 'sgd':
        p_score = clf.predict_proba(fvs_test)
        # print(metrics.roc_curve(target_test,p_score[:,0]))
        targets2d = np.array([1-target_test,target_test])
        targets2d = np.transpose(targets2d)

        # Compute ROC curve and ROC area for each class
        print(c[0])
        fpr[c[0]], tpr[c[0]], _ = roc_curve(targets2d[:, 0], p_score[:, 0])
        roc_auc[c[0]] = auc(fpr[c[0]], tpr[c[0]])

plt.figure()
print(fpr.keys())
lw = 2
cl = 'mn'
plt.plot(fpr[cl], tpr[cl], color='darkorange',
         lw=lw, label='Multinomial Naive Bayes (area = %0.2f)' % roc_auc[cl])

cl = 'bn'
plt.plot(fpr[cl], tpr[cl], color='red',
         lw=lw, label='Bernoulli Naive Bayes (area = %0.2f)' % roc_auc[cl])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')


cl = 'lr'
plt.plot(fpr[cl], tpr[cl], color='black',
         lw=lw, label='Logistic Regression (area = %0.2f)' % roc_auc[cl])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig("LaTeX/ROC.png")
plt.show()
