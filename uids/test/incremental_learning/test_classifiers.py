import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import pickle
import os
from sklearn import svm
from sklearn import linear_model

from sklearn.linear_model import SGDClassifier
import random

from sklearn.learning_curve import learning_curve

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        print filename
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return embeddings
    return None


def train_and_classify(class_data, outlier_data, clf):

    print "=====================================================\n"
    print "        EVALUATE CLASSIFIER\n"
    print "=====================================================\n"

    all_classes = [12,3,2,1,5,6]
    labels = np.repeat(2, len(class_data))
    start = time.time()
    clf.partial_fit(class_data, labels, all_classes)
    print "fitting took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_1 = clf.predict(class_data)
    print "prediction took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_2 = clf.predict(outlier_data)
    print "prediction took {} seconds".format(time.time()-start)

    print "{}/{} outliers have been detected".format((label_pred_2 < 0).sum(), len(outlier_data))
    print "{}/{} inliers have been detected".format((label_pred_1 > 0).sum(), len(class_data))


def train_both_and_classify(class_data, outlier_data, clf):

    print "=====================================================\n"
    print "        EVALUATE CLASSIFIER\n"
    print "=====================================================\n"

    all_classes = [12,3,2,1,5,6]
    labels = np.repeat(1, len(class_data))
    start = time.time()
    clf.partial_fit(class_data, labels, all_classes)
    print "fitting class data took {} seconds".format(time.time()-start)

    labels = np.repeat(2, len(outlier_data))
    start = time.time()
    clf.partial_fit(outlier_data, labels)
    print "fitting random data took {} seconds".format(time.time() - start)

    start = time.time()
    label_pred_1 = clf.predict(class_data)
    print "prediction took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_2 = clf.predict(outlier_data)
    print "prediction took {} seconds".format(time.time()-start)

    outlier_count = len(label_pred_2[label_pred_2==2])

    print "{}/{} outliers have been detected".format(outlier_count, len(outlier_data))
    print "{}/{} inliers have been detected".format(len(label_pred_1[label_pred_1==1]), len(class_data))


def batches(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def test_sgd(X, Y):
    from sklearn.linear_model import SGDClassifier
    import random
    clf2 = SGDClassifier(loss='log')  # shuffle=True is useless here
    shuffledRange = range(len(X))
    n_iter = 5
    for n in range(n_iter):
        random.shuffle(shuffledRange)
        shuffledX = [X[i] for i in shuffledRange]
        shuffledY = [Y[i] for i in shuffledRange]
        for batch in batches(range(len(shuffledX)), 10000):
            clf2.partial_fit(shuffledX[batch[0]:batch[-1] + 1], shuffledY[batch[0]:batch[-1] + 1],
                             classes=np.unique(Y))


# ================================= #
#              Main

if __name__ == '__main__':


    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_lfw.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")


    emb1_split = np.split(np.array(emb1), 2)

    sgda = linear_model.SGDClassifier()

    train_both_and_classify(emb1_split[0], emb2, sgda)


    label_pred = sgda.predict(emb1_split[1])

    print label_pred
    print "{}/{} outliers have been detected".format(len(label_pred[label_pred==1]), len(label_pred))