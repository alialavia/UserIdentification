import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import pickle
import os
from sklearn import svm
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
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
    if os.path.isfile(filename):
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
#           Test functions

def test_1():
    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_lfw.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")


    emb1_split = np.split(np.array(emb1), 2)

    sgda = linear_model.SGDClassifier()

    train_both_and_classify(emb1_split[0], emb2, sgda)

    label_pred = sgda.predict(emb1_split[1])

    print label_pred
    print "{}/{} outliers have been detected".format(len(label_pred[label_pred==1]), len(label_pred))


def test_incremental_learning_rate(clf, nr_batches=50, verbose=True, init_shuffle=True):
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # select ds
    target = emb2

    # prepare ds
    np.random.shuffle(target)
    while len(target) % nr_batches != 0:
        target = target[:-1]

    # split into train and test sets
    split_set = np.array_split(target, nr_batches)

    # select test set
    X_test = split_set[-1]

    # outlier dataset
    X_outliers = emb_lfw

    # plotting
    error_train = []
    error_test = []
    error_outliers = []

    for i in range(0, nr_batches-1):
        if verbose:
            print "=====================================================\n"
            print "        Training Round {}\n".format(i)
            print "=====================================================\n"

        X_train = split_set[i]

        # fit
        inlier_labels = np.repeat(1, len(X_train))

        if i == 0:
            all_classes = [0, 1]
            rest_labels = np.repeat(0, len(X_outliers))
            # mix samples with outliers
            x_2 = np.concatenate((X_train, X_outliers))
            y_2 = np.concatenate((inlier_labels, rest_labels))

            if init_shuffle is True:
                shuffledRange = range(len(y_2))
                random.shuffle(shuffledRange)
                x_2 = [x_2[i] for i in shuffledRange]
                y_2 = [y_2[i] for i in shuffledRange]

            clf.partial_fit(x_2, y_2, all_classes)
        else:
            clf.partial_fit(X_train, inlier_labels)

        # evaluate
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == 0].size
        n_error_test = y_pred_test[y_pred_test == 0].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

        error_train.append(n_error_train/float(len(X_train))*100.0)
        error_test.append(n_error_test/float(len(X_test))*100.0)
        error_outliers.append(n_error_outliers/ float(len(X_outliers))*100.0)

        # display results
        if verbose:
            print "error train: {}/{}, error novel regular: {}/{}, error novel abnormal: {}/{}" \
                .format(
                        n_error_train, len(X_train),
                        n_error_test, len(X_test),
                        n_error_outliers, len(X_outliers))
            print "error train: {:.2f}%, error novel regular: {:.2f}%, error novel abnormal: {:.2f}%" \
                .format(
                        n_error_train/float(len(X_train))*100.0,
                        n_error_test/float(len(X_test))*100.0,
                        n_error_outliers/ float(len(X_outliers))*100.0)
    # extract error
    fig = plt.figure()
    plt.plot(range(0, nr_batches-1), error_train, label="Training data")
    plt.plot(range(0, nr_batches-1), error_test, label="Test data")
    plt.plot(range(0, nr_batches-1), error_outliers, label="Outlier data")
    plt.legend()
    plt.xlabel('Training Batch')
    plt.ylabel('Detection Error Rate')
    plt.title('Detector drift')
    plt.show()


def test_compare_online_solvers():

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    heldout = [0.95, 0.90, 0.75, 0.50, 0.02, 0.001]
    rounds = 20
    xx = 1. - np.array(heldout)

    X_train = emb2
    X_outliers = emb_lfw

    # fit
    inlier_labels = np.repeat(1, len(X_train))
    rest_labels = np.repeat(0, len(X_outliers))

    ds = [emb1, emb2, emb3]

    X = []
    y = []
    for label, emb in enumerate(ds):
        if len(X) == 0:
            X = emb
            y = np.repeat(label, len(emb))
        else:
            X = np.concatenate((X, emb))
            y = np.concatenate((y, np.repeat(label, len(emb))))

    # choose classifiers
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

# ================================= #
#              Main

if __name__ == '__main__':
    # clf = SGDClassifier(loss='log')
    # clf = PassiveAggressiveClassifier(loss='squared_hinge', C=1.0)
    clf = Perceptron()
    test_incremental_learning_rate(clf, nr_batches=3)  # shuffle=True is useless here)

    # compare learning rate of different online solvers
    # test_compare_online_solvers()

