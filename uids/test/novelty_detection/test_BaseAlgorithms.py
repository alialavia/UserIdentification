from sklearn import svm
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.ensemble import IsolationForest

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


# ================================= #
#        Test Functions


def test_find_gamma():
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")



    # prepare ds
    np.random.shuffle(emb2)
    if len(emb2) % 2 != 0:
        emb2 = emb2[:-1]

    split_set = np.array_split(emb2, 2)

    X_train = split_set[0]
    X_test = split_set[1]
    X_outliers = emb_lfw

    for i in range(100):
        gamma = 0.001*i + 0.001

        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma)
        # train
        clf.fit(X_train)

        # evaluate
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

        # display results
        print "gamma: {}| error train: {}/{}, error novel regular: {}/{}, error novel abnormal: {}/{}"\
            .format(gamma,
                    n_error_train, len(X_train),
                    n_error_test, len(X_test),
                    n_error_outliers, len(X_outliers))

def test_1():
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # prepare ds
    np.random.shuffle(emb2)
    if len(emb2) % 2 != 0:
        emb2 = emb2[:-1]

    split_set = np.array_split(emb2, 2)

    X_train = split_set[0]
    X_test = split_set[1]
    X_outliers = emb_lfw

    for i in range(20):
        nu = 0.001*i + 0.001

        clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.15)
        # train
        clf.fit(X_train)

        # evaluate
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

        # display results
        print "nu: {}| error train: {}/{}, error novel regular: {}/{}, error novel abnormal: {}/{}"\
            .format(nu,
                    n_error_train, len(X_train),
                    n_error_test, len(X_test),
                    n_error_outliers, len(X_outliers))


def test_learning_rate(clf, nr_batches=50, verbose=False, init_shuffle=True):
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
    batch_size = len(target)/nr_batches

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
    training_time = []

    for i in range(1, nr_batches):
        if verbose:
            print "=====================================================\n"
            print "        Training Round {}\n".format(i)
            print "=====================================================\n"

        # select training set
        X_train = np.concatenate((split_set[0:i]))

        # shuffle
        if init_shuffle is True:
            random.shuffle(X_train)

        # fit classifier
        start = time.time()
        clf.fit(X_train)
        training_time.append(float(time.time()-start)*1000)

        # evaluate
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

        error_train.append(n_error_train / float(len(X_train)) * 100.0)
        error_test.append(n_error_test / float(len(X_test)) * 100.0)
        error_outliers.append(n_error_outliers / float(len(X_outliers)) * 100.0)

        # display results
        if verbose:
            print "error train: {}/{}, error novel regular: {}/{}, error novel abnormal: {}/{}" \
                .format(
                n_error_train, len(X_train),
                n_error_test, len(X_test),
                n_error_outliers, len(X_outliers))
            print "error train: {:.2f}%, error novel regular: {:.2f}%, error novel abnormal: {:.2f}%" \
                .format(
                n_error_train / float(len(X_train)) * 100.0,
                n_error_test / float(len(X_test)) * 100.0,
                n_error_outliers / float(len(X_outliers)) * 100.0)

    # extract error
    fig = plt.figure()
    plt.plot(range(0, nr_batches - 1), error_train, label="Training data")
    plt.plot(range(0, nr_batches - 1), error_test, label="Test data")
    plt.plot(range(0, nr_batches - 1), error_outliers, label="Outlier data")
    plt.plot(range(0, nr_batches - 1), training_time, label="Training Time [ms]")
    plt.legend()
    plt.xlabel('Batch size')
    plt.ylabel('Detection Error Rate')


# ================================= #
#              Main

if __name__ == '__main__':

    # select classifier


    # # RBF SVM
    # clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.15)
    # test_incremental_learning_rate(clf)
    # plt.title('Detector drift SVM[rbf]')
    #
    # # LINEAR SVM
    # clf = svm.OneClassSVM(nu=0.1, gamma=0.15)
    # test_incremental_learning_rate(clf)
    # plt.title('Detector drift SVM[linear]')

    # ISOLATION FOREST
    clf = IsolationForest(random_state=np.random.RandomState(42))
    test_learning_rate(clf, nr_batches=3, verbose=True)
    plt.title('Detector drift isolation forest')

    plt.show()
