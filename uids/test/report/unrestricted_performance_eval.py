import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
import csv
from numpy import random
from uids.online_learning.ABOD import ABOD
from sklearn import metrics
from uids.sklearn.Classifiers import *
from numpy.random import RandomState

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')  # path to the model directory

current_milli_time = lambda: int(round(time.time() * 1000))

def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def get_all_param_variants(pgrid):

    pairs = []

    if len(pgrid) == 1:
        for valp0 in pgrid[pgrid.keys()[0]]:
            pair = {}
            pair[pgrid.keys()[0]] = valp0
            pairs.append(pair)
    elif len(pgrid) == 2:
        for valp0 in pgrid[pgrid.keys()[0]]:
            for valp1 in pgrid[pgrid.keys()[1]]:
                pair = {}
                pair[pgrid.keys()[0]] = valp0
                pair[pgrid.keys()[1]] = valp1
                pairs.append(pair)
    elif len(pgrid) == 3:
        for v0 in pgrid[pgrid.keys()[0]]:
            for v1 in pgrid[pgrid.keys()[1]]:
                for v2 in pgrid[pgrid.keys()[2]]:
                    pair = {}
                    pair[pgrid.keys()[0]] = v0
                    pair[pgrid.keys()[1]] = v1
                    pair[pgrid.keys()[2]] = v2

                    pairs.append(pair)
    else:
        raise ValueError

    return pairs


def eval_unrestr_perf(clf, avg_cycles=10, nr_test_samples=160, filename=""):
    """

    :param clf:
    :param param_grid:
    :param avg_cycles:
    :param nr_training_samples:
    :param kfold: Nr folds for uncombined scene training.
    :param combine_scenes:
    :return:
    """


    save_csv = True
    randomize = True
    nr_iters = avg_cycles
    training_sizes = [5, 10, 20, 40]

    emb1 = load_embeddings("matthias/matthias_test.pkl")
    emb2 = load_embeddings("matthias/matthias_test2.pkl")
    emb_ul_unrestricted = load_embeddings("christian/christian_clean.pkl")

    # select scenes and outlier class
    class_ds1 = emb1
    class_ds2 = emb2
    outlier_ds_unrestricted = emb_ul_unrestricted

    if len(outlier_ds_unrestricted) < nr_test_samples/2:
        print "Not enough outlier samples!"
        return

    if len(class_ds2) < nr_test_samples/2:
        print "Not enough class test samples!"
        return

    # select amount of test data
    class_ds2 = class_ds2[0:nr_test_samples/2]
    outlier_ds_unrestricted = outlier_ds_unrestricted[0:nr_test_samples/2]

    clf_name = clf.__class__.__name__

    avg_precision = []
    avg_recall = []
    avg_f1_scores = []
    avg_training_time = []
    avg_prediction_time = []
    avg_youden_indices = []

    test_samples = np.concatenate((class_ds2, outlier_ds_unrestricted))
    prng = RandomState()

    # iterate over training set sizes
    for nr_training_samples in training_sizes:


        # allocate storage
        iter_precision = []
        iter_recall = []
        iter_f1_scores = []
        iter_youden_indices = []
        iter_training_time = []
        iter_prediction_time = []

        sys.stdout.write("Training with size {} ".format(nr_training_samples))
        start = time.time()
        for i in range(0, nr_iters):
            if i == 1:
                est_time = (time.time()-start) * nr_training_samples
                if est_time > 60:
                    est_time = est_time/60.0
                    sys.stdout.write(" | Estimated time: {:.2f} min, Iteration: ".format(est_time))
                else:
                    sys.stdout.write(" | Estimated time: {:.2f} sec, Iteration: ".format(est_time))
            if i > 0:
                sys.stdout.write("{}, ".format(i+1))

            # shuffle every round
            prng = RandomState(i+1)

            # select training samples
            training_data = prng.permutation(class_ds1)
            training_samples = training_data[0:nr_training_samples]

            # fit
            start = current_milli_time()
            clf.fit(training_samples)
            iter_training_time.append(current_milli_time() - start)

            # predict
            start = current_milli_time()
            labels_predicted = clf.predict(test_samples)
            iter_prediction_time.append(current_milli_time() - start)

            # calculate metrics
            true_nr_positives = nr_test_samples / 2
            true_nr_negatives = nr_test_samples / 2
            tp = np.count_nonzero(labels_predicted[0:true_nr_positives] == 1)
            fn = true_nr_positives - tp
            fp = np.count_nonzero(labels_predicted[true_nr_positives:] == 1)
            tn = true_nr_negatives - fp
            fpr = float(fp) / float(fp + tn)

            recall = float(tp) / float(tp + fn)
            try:
                precision = float(tp) / float(tp + fp)
                f1_score = 2 * float(precision * recall) / float(precision + recall)
            except ZeroDivisionError:
                precision = 0
                f1_score = 0

            iter_precision.append(precision)
            iter_recall.append(recall)
            iter_f1_scores.append(f1_score)
            iter_youden_indices.append(precision+recall-1)

        print "\n"

        avg_precision.append(np.mean(iter_precision))
        avg_recall.append(np.mean(iter_recall))
        avg_f1_scores.append(np.mean(iter_f1_scores))
        avg_training_time.append(np.mean(iter_training_time))
        avg_prediction_time.append(np.mean(iter_prediction_time))
        avg_youden_indices.append(np.mean(iter_youden_indices))


    if True:
        if filename == "":
            filename = clf_name+'_unrestr_accuracy.csv'

        with open(filename, 'wb') as csvfile:
            # write configuration of best results over multiple random tests
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Classifier", clf_name])
            writer.writerow(["Test set size", nr_test_samples])
            writer.writerow("")
            writer.writerow(["Training Set Size"] + training_sizes)
            writer.writerow(["Precision"] + ["%0.6f" % i for i in avg_precision])
            writer.writerow(["Recall"] + ["%0.6f" % i for i in avg_recall])
            writer.writerow(["F1"] + ["%0.6f" % i for i in avg_f1_scores])
            writer.writerow(["Youden-Index"] + ["%0.6f" % i for i in avg_youden_indices])
            writer.writerow("")
            writer.writerow(["Training-Time (s)"] + ["%0.6f" % i for i in avg_training_time])
            writer.writerow(["Predicition-Time (ms)"] + ["%0.6f" % i for i in avg_prediction_time])

            # writer.writerow(["Precision, Recall, Training-Time (s):"])
            # writer.writerow(["%0.6f" % i for i in iter_precision])
            # writer.writerow(["%0.6f" % i for i in iter_recall])
            # writer.writerow(["%0.6f" % i for i in iter_training_time])



# ================================= #
#              Main

if __name__ == '__main__':

    if True:
        clf = L2Estimator(T=0.99)
        eval_unrestr_perf(clf, avg_cycles=20, nr_test_samples=800, filename="")

    if True:
        clf = ABODEstimator(T=0.3)
        eval_unrestr_perf(clf, avg_cycles=20, nr_test_samples=800, filename="")