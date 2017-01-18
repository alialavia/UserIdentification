from sklearn import svm
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import *
import csv
from numpy import genfromtxt
from scipy.spatial import distance as dist

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None

def load_labels(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        my_data = genfromtxt(filename, delimiter=',')
        return my_data
    return None


class ABOD:
    data = None
    verbose = False
    # impl. : see https://github.com/MarinYoung4596/OutlierDetection

    def __init__(self):
        self.data = []

    def train(self, data):
        self.data = data

    def angle(self, x, y):
        costheta = 1 - pairwise_distances(x.reshape(1, -1), y.reshape(1, -1), metric='cosine')
        return np.arccos(costheta)

    def predict(self, samples, method="mean_cosdist"):
        if method == "angle":
            return self.__predict_angle(samples)
        elif method == "cosdist_var":
            return self.__predict_cosine_dist_var(samples)
        else:
            return self.__predict_mean_cosdist(samples)

    # ------------SAMPLE WEIGHTING--------------- #

    # ------------PREDICTION METRICS--------------- #

    def __predict_mean_cosdist(self, samples):
        cos_dist = pairwise_distances(samples, self.data, metric='cosine')
        result = np.mean(cos_dist, axis=1)
        print len(result)
        return result

    def __predict_angle(self, samples):

        dist = []
        for x in samples:
            angles = []
            for y in self.data:
                angles.append(self.angle(x,y))
            dist.append(angles)

        dist = np.array(dist)

        if self.verbose:
            print "--- Max cosine distance: {}".format(np.max(dist))
        result = np.var(dist, axis=1)
        if self.verbose:
            print "--- ABOD: {}".format(result)
        return result

    def __predict_cosine_dist_var(self, samples):
        cos_dist = pairwise_distances(samples, self.data, metric='cosine')
        if self.verbose:
            print "--- Max cosine distance: {}".format(np.max(cos_dist))
        result = np.var(cos_dist, axis=1)
        if self.verbose:
            print "--- ABOD: {}".format(result)
        return result


# ================================= #
#        Test Functions


def test_detection_rate(classifiers, nr_batches=50, ds_limit=500, verbose=False, init_shuffle=True, display=True):
    """
    user plt.show() at the end
    """
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias_big.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")


    # filter blurred images
    l = load_labels('blur_labels_matthias_big.csv')
    l = l[:,1]
    mask = l==0
    emb2 = emb2[mask]


    # select ds
    target = emb2

    # prepare ds
    np.random.shuffle(target)
    np.random.shuffle(emb_lfw)


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
    total_error_train = [None] * len(classifiers)
    total_error_test = [None] * len(classifiers)
    total_error_outliers = [None] * len(classifiers)
    total_training_time = [None] * len(classifiers)

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

        j = 0
        for clf_name, clf in classifiers:

            # fit classifier
            start = time.time()
            clf.fit(X_train)

            if i == 1:
                total_error_train[j] = []
                total_error_test[j] = []
                total_error_outliers[j] = []
                total_training_time[j] = []

            total_training_time[j].append(float(time.time()-start)*1000)

            # evaluate
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            y_pred_outliers = clf.predict(X_outliers)
            n_error_train = y_pred_train[y_pred_train == -1].size
            n_error_test = y_pred_test[y_pred_test == -1].size
            n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

            total_error_train[j].append(n_error_train / float(len(X_train)) * 100.0)
            total_error_test[j].append(n_error_test / float(len(X_test)) * 100.0)
            total_error_outliers[j].append(n_error_outliers / float(len(X_outliers)) * 100.0)

            j += 1

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

    # plot
    if display is True:
        j = 0
        for clf_name, clf in classifiers:
            # extract error
            fig = plt.figure()

            x_axis_values = range(1 * batch_size, nr_batches * batch_size, batch_size)

            plt.plot(x_axis_values, total_error_train[j], label="Training data")
            plt.plot(x_axis_values, total_error_test[j], label="Test data")
            plt.plot(x_axis_values, total_error_outliers[j], label="Outlier data")
            # plt.plot(range(0, nr_batches - 1), training_time, label="Training Time [ms]")
            plt.legend()
            plt.xlabel('Training Set Size')
            plt.ylabel('Detection Error Rate')
            plt.title('Learning Rate {}'.format(clf_name))
            j += 1

    return (total_error_train, total_error_test, total_error_outliers, total_training_time,batch_size)


def test_ABOD():

    clf = ABOD()

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_matthias_big.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")


    clf.train(emb2)
    abod_class = clf.predict(emb2)

    abod_class_2 = clf.predict(emb3)
    abod_outliers = clf.predict(emb_lfw)

    n, bins, patches = plt.hist(np.transpose(abod_class), 50, normed=1, facecolor='blue', alpha=0.75)
    n, bins, patches = plt.hist(np.transpose(abod_class_2), 50, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(np.transpose(abod_outliers), 50, normed=1, facecolor='red', alpha=0.75)
    plt.show()

# ================================= #
#              Main

if __name__ == '__main__':



    # select classifier
    # classifiers = [
    #     ('Isolation Forest (0.5% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.005)),
    #     ('Isolation Forest (1% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.01)),
    #     ('Isolation Forest (10% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.1)),
    #     # ('Isolation Forest (15% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.15)),
    #     ('One-Class SVM (RBF)', svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.15)),
    #     ('One-Class SVM (Linear)', svm.OneClassSVM(nu=0.1, gamma=0.15))
    # ]

    # perform test


    classifiers = [
        # ('Isolation Forest (0.5% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.005)),
        # ('Isolation Forest (1% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.01)),
        # ('Isolation Forest (5% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.05)),
        # ('Isolation Forest (10% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.1))
        # # ('Isolation Forest (15% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.15)),
        # ('One-Class SVM (RBF)', svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.15)),
        ('One-Class SVM (Linear)', svm.OneClassSVM(nu=0.01, gamma=0.15)),
        # ('One-Class SVM (Linear)', svm.OneClassSVM(nu=0.2, gamma=0.15)),
        # ('One-Class SVM (Linear)', svm.OneClassSVM(nu=0.3, gamma=0.15)),
        # ('One-Class SVM (Linear)', svm.OneClassSVM(nu=0.4, gamma=0.15))
    ]

    # test_detection_rate(classifiers, nr_batches=50, verbose=False, display=True)
    # plt.show()

    # test_detection_rate(classifiers, nr_batches=20)
    # plt.show()

    test_ABOD()