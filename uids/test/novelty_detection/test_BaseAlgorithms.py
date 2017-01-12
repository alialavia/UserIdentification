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
    # print filename
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


def test_detection_rate(classifiers, nr_batches=50, verbose=False, init_shuffle=True, display=True):
    """
    user plt.show() at the end
    """
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

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


# ================================= #
#      Combined Testing Functions

def test_repeated_detection_rate(classifiers, nr_tests=2, nr_batches=4, plot_target='all'):

    sum_error_train = []
    sum_error_test = []
    sum_error_outliers = []
    sum_error_training_time = []

    plot_targets = ['all', 'average']
    if plot_target not in plot_targets:
        print "Invalid Plot target. Choose betweenn {}".format(plot_targets)
        return

    for i in range(0, nr_tests):
        (total_error_train,
         total_error_test,
         total_error_outliers,
         total_training_time,
         batch_size) = test_detection_rate(classifiers, nr_batches=nr_batches, verbose=False, display=False)

        sum_error_train.append(np.array(total_error_train))
        sum_error_test.append(np.array(total_error_test))
        sum_error_outliers.append(np.array(total_error_outliers))
        sum_error_training_time.append(np.array(total_training_time))

        print "Test nr {} complete".format(i)

    # take max
    max_error_train = np.max(sum_error_train, axis=0)
    max_error_test = np.max(sum_error_test, axis=0)
    max_error_outliers = np.max(sum_error_outliers, axis=0)
    max_error_training_time = np.max(sum_error_training_time, axis=0)

    # plot
    if plot_target == 'all':
        j = 0
        for clf_name, clf in classifiers:

            fig = plt.figure()
            for i in range(0, nr_tests):

                # extract error
                x_axis_values = range(1 * batch_size, nr_batches * batch_size, batch_size)
                plt.plot(x_axis_values, sum_error_train[i][j], label="Training data", color='r')
                plt.plot(x_axis_values, sum_error_test[i][j], label="Test data", color='g')
                plt.plot(x_axis_values, sum_error_outliers[i][j], label="Outlier data", color='b')

                # plt.plot(x_axis_values, max_error_training_time[j], label="Outlier data")
                # plt.plot(range(0, nr_batches - 1), sum_error_training_time, label="Training Time [ms]")
                plt.xlabel('Training Set Size')
                plt.ylabel('Detection Error Rate')
                plt.title('Average Learning Rate [{}] for {} Randomized Tests'.format(clf_name, nr_tests))

            # plot legend
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            plt.legend(handles, labels, loc='best')
            j += 1

    if plot_target == 'average':
        # take average
        sum_error_train = np.mean(sum_error_train, axis=0)
        sum_error_test = np.mean(sum_error_test, axis=0)
        sum_error_outliers = np.mean(sum_error_outliers, axis=0)
        sum_error_training_time = np.mean(sum_error_training_time, axis=0)

        j = 0
        for clf_name, clf in classifiers:
            # extract error
            fig = plt.figure()
            x_axis_values = range(1 * batch_size, nr_batches * batch_size, batch_size)
            plt.plot(x_axis_values, sum_error_train[j], label="Training data", color='r')
            plt.plot(x_axis_values, sum_error_test[j], label="Test data", color='g')
            plt.plot(x_axis_values, sum_error_outliers[j], label="Outlier data", color='b')

            # plt.plot(x_axis_values, max_error_training_time[j], label="Outlier data")
            # plt.plot(range(0, nr_batches - 1), sum_error_training_time, label="Training Time [ms]")
            plt.legend()
            plt.xlabel('Training Set Size')
            plt.ylabel('Detection Error Rate')
            plt.title('Average Learning Rate [{}] for {} Randomized Tests'.format(clf_name, nr_tests))
            j += 1

    plt.show()


# ================================= #
#              Main

if __name__ == '__main__':

    # select classifier
    classifiers = [
        # ('Isolation Forest (0.5% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.005)),
        # ('Isolation Forest (1% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.01)),
        # ('Isolation Forest (10% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.1)),
        ('Isolation Forest (15% contamination)', IsolationForest(random_state=np.random.RandomState(42), contamination=0.15)),
        # svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.15),
        # svm.OneClassSVM(nu=0.1, gamma=0.15)
    ]

    # perform test
    test_repeated_detection_rate(classifiers, nr_tests=10, nr_batches=4, plot_target='all')

