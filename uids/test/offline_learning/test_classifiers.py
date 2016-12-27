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
from sklearn.svm import LinearSVC

from pandas import DataFrame
from sklearn import preprocessing

import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

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


def test_sgd_incremental():

    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_lfw.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")


    # preprocess data ----------------

    scaler = preprocessing.StandardScaler().fit(emb1)
    emb1 = scaler.transform(emb1)
    emb1_split = np.split(np.array(emb1), 2)
    emb2 = scaler.transform(emb2)
    emb3 = scaler.transform(emb3)
    emb3_split = np.split(np.array(emb3), 2)

    # -----------------

    clf = linear_model.SGDClassifier()

    all_classes = [12,3,2,1,5,6]
    clf.partial_fit(emb1_split[0], np.repeat(1, len(emb1_split[0])), all_classes)
    clf.partial_fit(emb2, np.repeat(2, len(emb2)))

    pred = clf.predict(emb1_split[1])
    print "{}/{} inliers have been detected".format(len(pred[pred==1]), len(pred))

    # predict novelty
    pred = clf.predict(emb3)
    print "{}/{} inliers have been detected".format(len(pred[pred==1]), len(pred))


    # train novelty
    clf.partial_fit(emb3_split[0], np.repeat(3, len(emb3_split[0])))

    # predict class
    pred = clf.predict(emb3_split[1])
    print "{}/{} inliers have been detected".format(len(pred[pred==3]), len(emb3_split[1]))


def add_class_col(data, class_id):
    nr_samples, dims = emb1.shape
    print emb1.shape

    # add class
    b = np.zeros((nr_samples, dims+1))

    print b.shape
    b[:, :-1] = class_id
    return b


# ================================= #
#              Main

if __name__ == '__main__':

    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_lfw.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")


    # preprocess data ----------------

    scaler = preprocessing.StandardScaler().fit(emb1)
    emb1 = scaler.transform(emb1)
    emb1_split = np.split(np.array(emb1), 2)
    emb2 = scaler.transform(emb2)
    emb3 = scaler.transform(emb3)
    emb3_split = np.split(np.array(emb3), 2)

    # -----------------


    test = add_class_col(emb1_split[0], 22)

    print test[1]

    nr_samples = emb1.shape[0]





    combined = np.concatenate((emb1_split[0],emb1_split[1]))
    # print combined

    # 2 -----------------------------------------------------------------------------

    print gibtsnied




    """
    ==================================
    Comparing various online solvers
    ==================================

    An example showing how different online solvers perform
    on the hand-written digits dataset.

    """
    # Author: Rob Zinkov <rob at zinkov dot com>
    # License: BSD 3 clause



    heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
    rounds = 20
    digits = datasets.load_digits()





    X, y = digits.data, digits.target

    print X

    classifiers = [
        ("SGD", SGDClassifier()),
        ("ASGD", SGDClassifier(average=True)),
        ("Perceptron", Perceptron()),
        ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                             C=1.0)),
        ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                              C=1.0)),
        ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
    ]

    xx = 1. - np.array(heldout)

    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        for i in heldout:
            yy_ = []
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        plt.plot(xx, yy, label=name)

    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    plt.show()
