import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import pickle
import os
from sklearn import svm
from sklearn import linear_model

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')  # path to the model directory


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def tune_OneClassSVM_deprecated():
    emb1 = load_embeddings("matthias_test.pkl")
    emb2 = load_embeddings("matthias_test2.pkl")
    emb3 = load_embeddings("embeddings_christian_clean.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")


    # Loading the Digits dataset
    digits = datasets.load_digits()
    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


    # --------------------

    # X_train = emb1
    # X_test = emb_lfw
    # # use 50 images for training and all for test
    # # X_train = X_train[0:50]
    # y_train = np.repeat(1, len(X_train))
    # y_test = np.repeat(-1, len(X_test))
    # # [0.005, 0.01, 0.02, 0.04, 0.08]

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10,1,1e-2, 1e-3, 1e-4],'nu': np.arange(0.005, 0.1, 0.001)},
                        # {'kernel': ['linear'],'gamma': [10,1,1e-2, 1e-3, 1e-4],'nu': [0.005, 0.01, 0.02, 0.04, 0.08]}
                        ]

    scores = ['precision', 'recall']

    for score in scores:

        print("")

        print "============================================="
        print(" Tuning hyper-parameters for %s" % score)
        print "=============================================\n"

        clf = GridSearchCV(svm.OneClassSVM(),  # SVC(C=1)
                           tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print "--------------------------------------------"
        print(clf.best_params_)
        print "--------------------------------------------\n"
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("mean: %0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print("Detailed classification report:")
        print("The model is trained on the full development set.")


        print "-------------------------TEST---------------------------"
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


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
        for valp0 in pgrid[pgrid.keys()[0]]:
            for valp1 in pgrid[pgrid.keys()[1]]:
                for valp2 in pgrid[pgrid.keys()[2]]:
                    pair = {}
                    pair[pgrid.keys()[0]] = valp0
                    pair[pgrid.keys()[1]] = valp1
                    pair[pgrid.keys()[2]] = valp2
                    pairs.append(pair)
    else:
        raise ValueError


    return pairs


def tune_OneClassSVM():
    emb1 = load_embeddings("matthias_test.pkl")
    emb2 = load_embeddings("matthias_test2.pkl")
    emb3 = load_embeddings("embeddings_christian_clean.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # PARAMETERS
    nr_training_samples = 50
    nr_splits = 4
    verbose = True

    testset = emb1
    outlier_set = emb_lfw

    if(nr_training_samples*nr_splits > len(testset)):
        print "Testset size {} too small ({})".format(len(testset), nr_training_samples*nr_splits)
        return

    class_samples = emb1[0:nr_training_samples*nr_splits]
    outliers = emb_lfw[0:(nr_splits-1)*nr_training_samples]
    kf = KFold(n_splits=nr_splits, shuffle=False)


    # for each parameter:
    # p_grid = [{'kernel': ['rbf'], 'gamma': [10,1,1e-2, 1e-3, 1e-4],'nu': np.arange(0.005, 0.1, 0.001)}]

    # param_grid = {'nu': np.arange(0.001, 0.5, 0.001), 'kernel': ['rbf']}
    param_grid = {'nu': [2,1], 'kernel': ['rbf']}

    precision_values = []
    recall_values = []


    for params_svm in get_all_param_variants(param_grid):
        # init classifiers
        clf = svm.OneClassSVM()
        clf.set_params(**params_svm)

        # build each parameter combination
        precision_scores = []
        recall_scores = []

        # calculate precision and recall in kfold cross validation
        for test_indices, train_indices in kf.split(class_samples):
            # print("%s %s" % (test_indices, train_indices))
            # print("%s %s" % (len(train_indices), len(test_indices)))

            # print "-----------------------------"

            # fit
            clf.fit(class_samples[train_indices])

            # build test set
            test_with_outliers = np.concatenate((class_samples[test_indices], outliers))
            labels = np.concatenate((np.repeat(1, len(test_indices)), np.repeat(-1, len(outliers))))

            # predict
            labels_predicted = clf.predict(test_with_outliers)

            # print "Classifying {} samples: {} are inliers".format(len(test_with_outliers), len(test_indices))

            # calculate metrics
            tp = np.count_nonzero(labels_predicted[0:len(test_indices)] == 1)
            fn = len(test_indices)-tp
            fp = np.count_nonzero(labels_predicted[len(test_indices):] == 1)
            tn = len(outliers)-fp
            fpr = float(fp)/float(fp+tn)
            f1_score = 2*float(tp)/float(2*tp+fp+fn)

            try:
                precision = float(tp) / float(tp + fp)
            except ZeroDivisionError:
                precision = 0

            recall = float(tp)/float(tp+fn)

            # print "tp: {}, tn: {}    ||   fn: {}, fp: {}, ".format(tp, fn, fp, tn)
            # print "precision: {}     ||   recall: {} ".format(precision, recall)

            precision_scores.append(precision)
            recall_scores.append(recall)

        # average precision and recall values

        precision_avg = np.mean(precision_scores)
        recall_avg = np.mean(recall_scores)

        precision_values.append(precision_avg)
        recall_values.append(recall_avg)

        if verbose:
            print "______________________________________________________________________\n" \
                  "Params: {}".format(params_svm)
            print "Precision: {}     ||     Recall: {}".format(precision_avg, recall_avg)

    # --------------- END EXPERIMENT

    print list(precision_values)
    print list(recall_values)

# ================================= #
#              Main

if __name__ == '__main__':
    tune_OneClassSVM()


