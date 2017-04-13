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


def tune_classifier(clf, param_grid, avg_cycles=10, nr_training_samples=50, kfold=4, combine_scenes=True):
    # param_grid = {'nu': np.arange(0.001, 0.1, 0.001), 'kernel': ['rbf']}
    # clf = svm.OneClassSVM()

    emb1 = load_embeddings("matthias_test.pkl")
    emb2 = load_embeddings("matthias_test2.pkl")
    emb3 = load_embeddings("embeddings_christian_clean.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")


    # tune = svm.OneClassSVM(kernel='linear')
    # tune.fit(emb1)
    # emb1 = emb1*tune.coef_


    # PARAMETERS
    nr_splits = kfold
    verbose = False
    save_csv = True
    randomize = True
    rdm_avg_cycles = avg_cycles
    objective = 'f1'

    if objective not in {'f1', 'youden'}:
        raise ValueError

    class_ds1 = emb1
    class_ds2 = emb2
    outlier_ds = emb_lfw
    clf_name = clf.__class__.__name__

    if randomize:
        nr_iters = rdm_avg_cycles
    else:
        nr_iters = 1

    iter_precision = []
    iter_recall = []
    iter_f1_scores = []
    iter_params = []
    iter_training_time = []

    for i in range(0, nr_iters):

        # shuffle same every time
        random.seed(i)  # Reset random state
        random.shuffle(class_ds1)
        random.shuffle(class_ds2)

        # combine the two scene datasets
        if combine_scenes:
            num_samples_each = np.max([len(class_ds1), len(class_ds2)])
            class_ds_combined = np.concatenate((class_ds1[0:num_samples_each], class_ds2[0:num_samples_each]))
        else:
            class_ds_combined = class_ds1
        # shuffle
        random.shuffle(class_ds_combined)

        if(nr_training_samples*nr_splits > len(class_ds_combined)):
            print "Testset size {} too small ({})".format(len(class_ds_combined), nr_training_samples*nr_splits)
            return

        class_samples = class_ds_combined[0:nr_training_samples*nr_splits]
        outliers = outlier_ds[0:(nr_splits-1)*nr_training_samples]
        kf = KFold(n_splits=nr_splits, shuffle=False)

        precision_values = []
        recall_values = []
        youden_index = []
        f1_scores = []
        training_time = []

        param_combinations = get_all_param_variants(param_grid)

        for i_param, clf_params in enumerate(param_combinations):
            # init classifiers
            clf.set_params(**clf_params)

            # build each parameter combination
            precision_scores_config = []
            recall_scores_config = []
            f1_scores_config = []
            training_time_config = []

            # calculate precision and recall in kfold cross validation
            for test_indices, train_indices in kf.split(class_samples):
                # print("%s %s" % (test_indices, train_indices))
                # print("%s %s" % (len(train_indices), len(test_indices)))

                # print "-----------------------------"
                # fit
                if clf_name in {'OneClassSVM', 'IsolationForest', 'ABOD'}:
                    start = time.time()
                    clf.fit(class_samples[train_indices])
                    training_time_config.append(time.time()-start)
                else:
                    print "Classifier {} not supported".format(clf_name)
                    raise Exception

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


                try:
                    precision = float(tp) / float(tp + fp)
                    f1_score = 2 * float(precision * recall) / float(precision + recall)
                except ZeroDivisionError:
                    precision = 0
                    f1_score = 0

                recall = float(tp)/float(tp+fn)



                # print "tp: {}, tn: {}    ||   fn: {}, fp: {}, ".format(tp, fn, fp, tn)
                # print "precision: {}     ||   recall: {} ".format(precision, recall)

                precision_scores_config.append(precision)
                recall_scores_config.append(recall)
                f1_scores_config.append(f1_score)

            # average precision and recall values
            precision_avg = np.mean(precision_scores_config)
            recall_avg = np.mean(recall_scores_config)
            training_time_avg = np.mean(training_time_config)
            f1_scores_avg = np.mean(f1_scores_config)

            precision_values.append(precision_avg)
            recall_values.append(recall_avg)
            youden_index.append(precision_avg+recall_avg-1)
            training_time.append(training_time_avg)
            f1_scores.append(f1_scores_avg)

            if verbose:
                print "______________________________________________________________________\n" \
                      "Params: {}".format(clf_params)
                print "Precision: {}     ||     Recall: {}".format(precision_avg, recall_avg)

        # --------------- END RANDOMIZED EXPERIMENT

        # print list(precision_values)
        # print list(recall_values)

        # --------------- BEST PARAMETERS

        if objective == 'f1':
            best_index = np.argmax(f1_scores)
        elif objective == 'youden':
            best_index = np.argmax(youden_index)

        best_params = param_combinations[best_index]
        print "_______________________________________________________"
        print "Best parameters (Youden-Index {}, F1: {}): {}".format(np.max(youden_index), np.max(f1_scores), best_params)
        print "Precision: {}     ||     Recall: {}".format(precision_values[best_index], recall_values[best_index])
        iter_precision.append(precision_values[best_index])
        iter_recall.append(recall_values[best_index])
        iter_f1_scores.append(f1_scores[best_index])
        iter_params.append(best_params)
        iter_training_time.append(training_time[best_index])

    # --------------- END RANDOM SERIES

    print "_______________________________________________________\n\n\n"
    print "                    FINAL EVALUATION:\n"
    print "Precision: ", iter_precision
    print "Precision Avg, std: {} +- {}".format(np.mean(iter_precision), 2*np.std(iter_precision))
    print "Recall: ", iter_recall
    print "Recall Avg, std: {} +- {}".format(np.mean(iter_recall), 2 * np.std(iter_recall))
    print "F1 score: {}", iter_f1_scores
    print "Parameters: ", iter_params

    if save_csv:
        # keep only best

        with open(clf_name+'_accuracy.csv', 'wb') as csvfile:
            # write configuration of best results over multiple random tests
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Set mixing : {}".format(np.mean(iter_precision), 2*np.std(iter_precision))])
            writer.writerow(["Precision Avg, std: {} +- {}".format(np.mean(iter_precision), 2*np.std(iter_precision))])
            writer.writerow(["Recall Avg, std: {} +- {}".format(np.mean(iter_recall), 2 * np.std(iter_recall))])
            writer.writerow(["F1 score, std: {} +- {}".format(np.mean(iter_f1_scores), 2 * np.std(iter_f1_scores))])
            writer.writerow(["Batch size: training: {}, prediction: {}".format(nr_training_samples, nr_training_samples*(nr_splits-1)*2)])


            writer.writerow("")
            writer.writerow(iter_precision)
            writer.writerow(iter_recall)
            writer.writerow(iter_training_time)

            if clf_name == 'OneClassSVM':
                writer.writerow(["Nu:"])
                nus = [tmp['nu'] for tmp in iter_params]
                writer.writerow(nus)
            else:
                writer.writerow(iter_params)
            writer.writerow("")
            # roc curve - save all precision and recall values


# ================================= #
#              Main

if __name__ == '__main__':

    if True:
        params_svm = {'nu': np.arange(0.001, 0.1, 0.001), 'kernel': ['rbf']}
        clf = svm.OneClassSVM()
        tune_classifier(clf, params_svm, avg_cycles=5)

    if False:
        params_if = {'contamination': np.arange(0.005, 0.1, 0.005)}
        clf = IsolationForest(random_state=np.random.RandomState(42))
        tune_classifier(clf, params_if)

    if False:
        params_abod = {'uncertainty_bandwidth': [0], 'threshold': np.arange(0.05, 0.5, 0.05)}
        clf = ABOD()
        tune_classifier(clf, params_abod, avg_cycles=2)
