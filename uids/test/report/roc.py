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

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')  # path to the model directory


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



def ROC(clf):

    # PARAMETERS
    nr_training_samples = 5
    nr_test_samples = 400

    save_csv = True
    combine_scenes = False

    # ---------------------------------------------

    emb0 = load_embeddings("embeddings_matthias.pkl")
    emb1 = load_embeddings("matthias_test.pkl")
    emb2 = load_embeddings("matthias_test2.pkl")
    emb3 = load_embeddings("embeddings_christian_clean.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    class_ds1 = emb1
    class_ds2 = emb2
    outlier_ds = emb_lfw

    # combine the two scene datasets
    if combine_scenes:
        num_samples_each = np.max([len(class_ds1), len(class_ds2)])
        class_ds_combined = np.concatenate((class_ds1[0:num_samples_each], class_ds2[0:num_samples_each]))
    else:
        class_ds_combined = class_ds1
    # shuffle
    random.shuffle(class_ds_combined)

    # ---------------------------------------------

    # fit
    # clf = svm.OneClassSVM(kernel='linear')
    clf = ABOD()
    clf.fit(class_ds_combined[0:nr_training_samples])

    # true labels
    labels = np.concatenate((np.repeat(1, nr_test_samples/2), np.repeat(2, nr_test_samples/2)))

    test_samples = np.concatenate((emb2[0:nr_test_samples/2], emb3[0:nr_test_samples/2]))

    # scores which are thresholded
    scores = clf.decision_function(test_samples)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc_val = auc(fpr, tpr)

    # ---------------------------------------------

    print "AUC: {}".format(auc_val)
    print "tpr: ", tpr
    print "fpr: ", fpr
    print "thresholds: ", thresholds

    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    # print "Precision: ", precision
    # print "Recall: ", recall

    plt.plot(recall, precision)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()

# ================================= #
#              Main

if __name__ == '__main__':

    clf = ABOD()
    ROC(clf)