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
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples', 'benchmark')  # path to the model directory

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


def eval_unrestr_perf(clf, persons=[]):

    if clf.__class__.__name__ in ["L2Estimator", "ABODEstimator"]:
        filename = clf.__class__.__name__ + "_T_" + str(clf.T) + '_multipeople_benchmark.csv'
    else:
        filename = clf.__class__.__name__ + '_multipeople_benchmark.csv'

    training_sizes = np.arange(3, 12)
    # training_sizes = [7]
    # training_sizes = [5,]

    with open(filename, 'wb') as csvfile:
        # write configuration of best results over multiple random tests
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Multi-People Classifier Benchmark"])
        writer.writerow([""])
        writer.writerow(["Classifier", clf.__class__.__name__])
        if clf.__class__.__name__ in ["L2Estimator", "ABODEstimator"]:
            writer.writerow(["Threshold", clf.T])
        writer.writerow("")

    for person in persons:

        emb_il = load_embeddings("{}.pkl".format(person))
        emb_ul = load_embeddings("{}_others.pkl".format(person))

        # if emb_il == None or emb_ul == None:
        #     raise Exception

        if len(emb_il) < 10:
            raise Exception

        print "Inliers: {}, Outliers: {}".format(len(emb_il), len(emb_ul))

        precision_values = []
        recall_values = []
        f1_scores = []
        training_time = []
        prediction_time = []
        youden_indices = []
        area_under_curves = []

        for t_size in training_sizes:

            training_samples = emb_il[0:t_size]

            # test samples
            test_samples = np.concatenate((emb_il[t_size:], emb_ul))
            nr_positives = len(emb_il) - t_size     # nr positives in test samples

            true_labels = np.concatenate((np.repeat(1, nr_positives), np.repeat(-1, len(emb_ul))))

            # print nr_positives
            # print len(training_samples)
            # print len(emb_il[t_size:])


            # print training_samples[0:5,1]
            # print test_samples[7:10,1]

            # fit
            start = current_milli_time()
            clf.fit(training_samples)
            training_time.append(current_milli_time() - start)

            # predict
            start = current_milli_time()
            labels_predicted = clf.predict(test_samples)
            prediction_time.append(current_milli_time() - start)

            # roc metrics
            scores = clf.decision_function(test_samples)    # scores which are thresholded

            if clf.__class__.__name__ == 'L2Estimator':
                scores = 20 - scores

            # print ["%0.3f" % i for i in scores]


            # print classification_report(true_labels, labels_predicted)
            # print roc_auc_score(true_labels, labels_predicted)
            # print roc_auc_score(true_labels, scores)

            # print scores
            fpr, tpr, thresholds = metrics.roc_curve(true_labels, scores, pos_label=1)
            # print thresholds
            auc_val = auc(fpr, tpr)
            area_under_curves.append(auc_val)
            # print "fpr: {}, tpr: {}".format(fpr, tpr)
            # print true_labels

            # calculate metrics
            tp = np.count_nonzero(labels_predicted[0:nr_positives] == 1)
            fn = nr_positives - tp
            fp = np.count_nonzero(labels_predicted[nr_positives:] == 1)
            tn = nr_positives - fp



            try:
                recall = float(tp) / float(tp + fn)
                precision = float(tp) / float(tp + fp)
                f1_score = 2 * float(precision * recall) / float(precision + recall)
            except ZeroDivisionError:
                precision = 0
                recall = 0
                f1_score = 0

            precision_values.append(precision)
            recall_values.append(recall)
            f1_scores.append(f1_score)
            youden_indices.append(precision + recall - 1)

        # save results
        # print precision_values
        # print recall_values

        if True:
            # append results
            with open(filename, 'ab') as csvfile:
                # write configuration of best results over multiple random tests
                writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(["Person", person])
                writer.writerow(["Training Set Size"] + list(training_sizes))
                writer.writerow(["Precision"] + ["%0.6f" % i for i in precision_values])
                writer.writerow(["Recall"] + ["%0.6f" % i for i in recall_values])
                writer.writerow(["F1"] + ["%0.6f" % i for i in f1_scores])
                writer.writerow(["Youden-Index"] + ["%0.6f" % i for i in youden_indices])
                writer.writerow(["AUC"] + ["%0.3f" % i for i in area_under_curves])
                writer.writerow("")

# ================================= #
#              Main

if __name__ == '__main__':

    # person = "ali"
    # person = "christian"
    # person = "matthias"
    person = "madleina"
    # person = "michael"
    # person = "tanu"
    # person = "laia"

    # good:
    # tanu,matthias, madleina, christian (bad), ali

    # bad:
    # laia,michael


    persons = ["ali", "christian", "matthias", "madleina", "michael", "tanu", "laia"]

    if False:
        print "-----------------------------"
        print "         L2\n\n"
        clf = L2Estimator(T=0.99)
        eval_unrestr_perf(clf, persons=persons)

    if True:
        print "-----------------------------"
        print "         ABOD\n\n"
        clf = ABODEstimator(T=2)
        eval_unrestr_perf(clf, persons=persons)