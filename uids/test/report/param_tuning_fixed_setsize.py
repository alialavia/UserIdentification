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


def tune_classifier(clf, param_grid, avg_cycles=10, nr_training_samples=50, nr_test_samples=160, combine_scenes=False, filename=""):
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
    objective = 'f1'

    emb1 = load_embeddings("matthias_test.pkl")
    emb2 = load_embeddings("matthias_test2.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # select scenes and outlier class
    class_ds1 = emb1
    class_ds2 = emb2
    outlier_ds = emb_lfw
    clf_name = clf.__class__.__name__

    # calculate folds
    if combine_scenes:
        nr_splits = float(nr_test_samples / (2. * nr_training_samples)) + 1
    else:
        nr_splits = float(nr_test_samples / (4. * nr_training_samples)) + 1

    if not nr_splits.is_integer():
        print "Invalid number of samples. Producing {} splits.".format(nr_splits)
        min_nr_test = nr_training_samples*2 if combine_scenes else nr_training_samples*4
        print "Adjust nr. training samples. E.g. {}, {}, {}, ...".format(min_nr_test, 2*min_nr_test, 3*min_nr_test)
        return

    nr_splits = int(nr_splits)
    print "Performing {}-fold cross-validation...".format(nr_splits)

    if objective not in {'f1', 'youden'}:
        raise ValueError

    # allocate storage
    iter_precision = []
    iter_recall = []
    iter_f1_scores = []
    iter_params = []
    iter_training_time = []
    iter_prediction_time = []
    iter_youden_indices = []

    for i in range(0, nr_iters):

        # shuffle same every time
        random.seed(i)  # Reset random state
        random.shuffle(class_ds1)
        random.shuffle(class_ds2)
        random.shuffle(outlier_ds)

        kf = KFold(n_splits=nr_splits, shuffle=False)
        param_combinations = get_all_param_variants(param_grid)

        # allocate metrics
        precision_values = []
        recall_values = []
        youden_index = []
        f1_scores = []
        training_time = []
        prediction_time = []

        # mode selection
        if combine_scenes:
            # -------------------- Case B: Train on 1 and 2, test on 1 and 2
            if (nr_training_samples/2+nr_test_samples/4) > len(class_ds1) or (nr_training_samples/2+nr_test_samples/4) > len(class_ds2):
                print "Too few samples!"
                return
        else:
            # -------------------- Case A: Train on 1, test on 1 and 2
            if (nr_training_samples+nr_test_samples/4) > len(class_ds1) or nr_test_samples/4 > len(class_ds2):
                print "Too few samples!"
                return

        for i_param, clf_params in enumerate(param_combinations):
            # init classifiers
            clf.set_params(**clf_params)

            # build each parameter combination
            precision_scores_config = []
            recall_scores_config = []
            f1_scores_config = []
            training_time_config = []
            prediction_time_config = []

            # -------------------- Case A: Train on 1, test on 1 and 2

            if combine_scenes:

                scene1_samples = class_ds1[0:(nr_training_samples/2+nr_test_samples/4)]
                scene2_samples = class_ds2[0:(nr_training_samples/2+nr_test_samples/4)]

                # calculate precision and recall in kfold cross validation
                for test_indices, train_indices in kf.split(scene1_samples):

                    training_samples = np.concatenate((scene1_samples[train_indices], scene2_samples[train_indices]))

                    # fit
                    if clf_name in {'OneClassSVM', 'IsolationForest', 'ABOD'}:
                        start = current_milli_time()
                        clf.fit(training_samples)
                        training_time_config.append(current_milli_time()-start)
                    else:
                        print "Classifier {} not supported".format(clf_name)
                        raise Exception

                    # build test set, add scene 2 , add outlier dataset
                    test_with_outliers = np.concatenate((scene1_samples[test_indices], scene2_samples[test_indices], outlier_ds[0:(nr_test_samples/2)]))
                    # 1/2 class, 1/2 outliers
                    labels = np.concatenate((np.repeat(1, nr_test_samples/2), np.repeat(-1, nr_test_samples/2)))

                    # predict
                    start = current_milli_time()
                    labels_predicted = clf.predict(test_with_outliers)
                    prediction_time_config.append(current_milli_time() - start)

                    # validate
                    if len(test_with_outliers) != nr_test_samples:
                        print "001: Check your code!"
                        return

                    # calculate metrics
                    true_nr_positives = nr_test_samples/2
                    true_nr_negatives = nr_test_samples/2
                    tp = np.count_nonzero(labels_predicted[0:true_nr_positives] == 1)
                    fn = true_nr_positives-tp
                    fp = np.count_nonzero(labels_predicted[true_nr_positives:] == 1)
                    tn = true_nr_negatives-fp
                    fpr = float(fp)/float(fp+tn)

                    recall = float(tp) / float(tp + fn)
                    try:
                        precision = float(tp) / float(tp + fp)
                        f1_score = 2 * float(precision * recall) / float(precision + recall)
                    except ZeroDivisionError:
                        precision = 0
                        f1_score = 0

                    # validate
                    if (tp + fn != nr_test_samples/2) or (fp + tn != nr_test_samples/2):
                        print "002: Check your code!"
                        print "tp: {}, tn: {}    ||   fn: {}, fp: {}, ".format(tp, fn, fp, tn)
                        print "precision: {}     ||   recall: {} ".format(precision, recall)
                        return

                    precision_scores_config.append(precision)
                    recall_scores_config.append(recall)
                    f1_scores_config.append(f1_score)

            else:

                class_samples_s1 = class_ds1[0:(nr_training_samples+nr_test_samples/4)]

                # calculate precision and recall in kfold cross validation
                for test_indices, train_indices in kf.split(class_samples_s1):

                    # fit
                    if clf_name in {'OneClassSVM', 'IsolationForest', 'ABOD'}:
                        start = current_milli_time()
                        clf.fit(class_samples_s1[train_indices])
                        training_time_config.append(current_milli_time()-start)
                    else:
                        print "Classifier {} not supported".format(clf_name)
                        raise Exception

                    # build test set, add scene 2 , add outlier dataset
                    test_with_outliers = np.concatenate((class_samples_s1[test_indices], class_ds2[0:nr_test_samples/4], outlier_ds[0:(nr_test_samples/2)]))
                    # 1/2 class, 1/2 outliers
                    labels = np.concatenate((np.repeat(1, nr_test_samples/2), np.repeat(-1, nr_test_samples/2)))

                    # predict
                    start = current_milli_time()
                    labels_predicted = clf.predict(test_with_outliers)
                    prediction_time_config.append(current_milli_time() - start)

                    # validate
                    if len(test_with_outliers) != nr_test_samples:
                        print "001: Check your code!"

                    # calculate metrics
                    true_nr_positives = nr_test_samples/2
                    true_nr_negatives = nr_test_samples/2
                    tp = np.count_nonzero(labels_predicted[0:true_nr_positives] == 1)
                    fn = true_nr_positives-tp
                    fp = np.count_nonzero(labels_predicted[true_nr_positives:] == 1)
                    tn = true_nr_negatives-fp
                    fpr = float(fp)/float(fp+tn)

                    recall = float(tp) / float(tp + fn)
                    try:
                        precision = float(tp) / float(tp + fp)
                        f1_score = 2 * float(precision * recall) / float(precision + recall)
                    except ZeroDivisionError:
                        precision = 0
                        f1_score = 0

                    # validate
                    if (tp + fn != nr_test_samples/2) or (fp + tn != nr_test_samples/2):
                        print "002: Check your code!"
                        print "tp: {}, tn: {}    ||   fn: {}, fp: {}, ".format(tp, fn, fp, tn)
                        print "precision: {}     ||   recall: {} ".format(precision, recall)
                        return

                    precision_scores_config.append(precision)
                    recall_scores_config.append(recall)
                    f1_scores_config.append(f1_score)



            # average precision and recall values
            precision_avg = np.mean(precision_scores_config)
            recall_avg = np.mean(recall_scores_config)
            training_time_avg = np.mean(training_time_config)
            prediction_time_avg = np.mean(prediction_time_config)
            f1_scores_avg = np.mean(f1_scores_config)

            precision_values.append(precision_avg)
            recall_values.append(recall_avg)
            youden_index.append(precision_avg+recall_avg-1)
            training_time.append(training_time_avg)
            prediction_time.append(prediction_time_avg)
            f1_scores.append(f1_scores_avg)

            # if verbose:
            #     print "______________________________________________________________________\n" \
            #           "Params: {}".format(clf_params)
            #     print "Precision: {}     ||     Recall: {}".format(precision_avg, recall_avg)

        # --------------- END RANDOMIZED EXPERIMENT

        # print list(precision_values)
        # print list(recall_values)

        # --------------- BEST PARAMETERS

        if objective == 'f1':
            best_index = np.argmax(f1_scores)
        elif objective == 'youden':
            best_index = np.argmax(youden_index)

        best_params = param_combinations[best_index]
        print "________________________{}/{}_______________________________".format(i+1, nr_iters)
        print "Best parameters (Youden-Index {:.2f}, F1: {:.2f}): {}".format(np.max(youden_index), np.max(f1_scores), best_params)
        print "Precision: {:.2f}     ||     Recall: {:.2f}".format(precision_values[best_index], recall_values[best_index])
        iter_precision.append(precision_values[best_index])
        iter_recall.append(recall_values[best_index])
        iter_f1_scores.append(f1_scores[best_index])
        iter_youden_indices.append(youden_index[best_index])
        iter_params.append(best_params)
        iter_training_time.append(training_time[best_index])
        iter_prediction_time.append(prediction_time[best_index])

    # --------------- END RANDOM SERIES

    print "_______________________________________________________\n\n\n"
    print "                    FINAL EVALUATION:\n"
    print "LEARNER: {}".format(clf_name)
    print "MODE: {}".format('Mixed Scene Training' if combine_scenes else 'Single Scene Training')
    print "K-FOLD VALIDATION: {} folds".format(nr_splits)
    print "-------------------------------------------------------"
    if combine_scenes:
        print "Batch size training: {} ({} S1/{} S2)".format(len(train_indices)*2, len(train_indices), len(train_indices))
    else:
        print "Batch size training: {} ".format(len(train_indices))
    print "Batch size test: {} ({} class, {} outliers)".format(len(labels), len(labels[labels==1]), len(labels[labels==-1]))
    print "_______________________________________________________"
    # print "Batch size: training: {}, prediction: {}".format(nr_training_samples, nr_training_samples * (nr_splits - 1) * 2)
    # print "Batch size training: {} (class)".format(len(class_training_samples))


    print "Precision Avg, std: {:.4f} +- {:.4f}".format(np.mean(iter_precision), 2*np.std(iter_precision))
    print "Recall Avg, std: {:.4f} +- {:.4f}".format(np.mean(iter_recall), 2 * np.std(iter_recall))
    print "Precision: ", ["%0.2f" % i for i in iter_precision]
    print "Recall: ", ["%0.2f" % i for i in iter_recall]
    print "F1 score: ", ["%0.2f" % i for i in iter_f1_scores]
    print "Parameters: ", iter_params

    if save_csv:
        # keep only best

        if filename == "":
            filename = clf_name+'_accuracy.csv'

        with open(filename, 'wb') as csvfile:
            # write configuration of best results over multiple random tests
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # settings
            if clf_name == 'OneClassSVM':
                writer.writerow(["LEARNER: {} ({})".format(clf_name, iter_params[0]['kernel'])])
            else:
                writer.writerow(["LEARNER: {}".format(clf_name)])
            writer.writerow(["MODE: {}".format('Mixed Scene Training' if combine_scenes else 'Single Scene Training')])
            writer.writerow(["K-FOLD VALIDATION: {} folds".format(nr_splits)])
            writer.writerow(["RANDOM ITERATIONS: {}".format(nr_iters)])
            writer.writerow(["Batch size: training: {}, test: {}".format(nr_training_samples, nr_test_samples)])
            writer.writerow(["Precision Avg, std: {} +- {}".format(np.mean(iter_precision), 2*np.std(iter_precision))])
            writer.writerow(["Recall Avg, std: {} +- {}".format(np.mean(iter_recall), 2 * np.std(iter_recall))])
            writer.writerow(["F1 score, std: {} +- {}".format(np.mean(iter_f1_scores), 2 * np.std(iter_f1_scores))])
            writer.writerow(["Youden index, std: {} +- {}".format(np.mean(iter_youden_indices), 2 * np.std(iter_youden_indices))])
            writer.writerow("")
            if clf_name == 'OneClassSVM':
                writer.writerow(["Train", "Test", "Folds", "Nu Median", "Nu Mean", "Nu Std", "P", "P std.", "R", "R std", "F1", "F1 std", "Youdens", "Youdens std", "Training Time", "Trainig Time std", "Prediction Time", "Prediction Time std"])
                writer.writerow([
                                 nr_training_samples, nr_test_samples, nr_splits, np.median([tmp['nu'] for tmp in iter_params]), np.mean([tmp['nu'] for tmp in iter_params]), np.std([tmp['nu'] for tmp in iter_params]),
                                 np.mean(iter_precision), np.std(iter_precision), np.mean(iter_recall), np.std(iter_recall),
                                 np.mean(iter_f1_scores), np.std(iter_f1_scores),
                                 np.mean(iter_youden_indices), np.std(iter_youden_indices),
                                 np.mean(iter_training_time), np.std(iter_training_time),
                                 np.mean(iter_prediction_time), np.std(iter_prediction_time)
                                 ])
            if clf_name == 'IsolationForest':
                writer.writerow(["Train", "Test", "Folds", "Cont Median", "Cont Mean", "Cont Std", "P", "P std.", "R", "R std", "F1", "F1 std", "Youdens", "Youdens std", "Training Time", "Trainig Time std", "Prediction Time", "Prediction Time std"])
                writer.writerow([
                                 nr_training_samples, nr_test_samples, nr_splits, np.median([tmp['contamination'] for tmp in iter_params]), np.mean([tmp['contamination'] for tmp in iter_params]), np.std([tmp['contamination'] for tmp in iter_params]),
                                 np.mean(iter_precision), np.std(iter_precision), np.mean(iter_recall), np.std(iter_recall),
                                 np.mean(iter_f1_scores), np.std(iter_f1_scores),
                                 np.mean(iter_youden_indices), np.std(iter_youden_indices),
                                 np.mean(iter_training_time), np.std(iter_training_time),
                                 np.mean(iter_prediction_time), np.std(iter_prediction_time)
                                 ])

            writer.writerow("")
            writer.writerow(["Precision, Recall, Training-Time (s):"])
            writer.writerow(["%0.6f" % i for i in iter_precision])
            writer.writerow(["%0.6f" % i for i in iter_recall])
            writer.writerow(["%0.6f" % i for i in iter_training_time])

            if clf_name == 'OneClassSVM':
                writer.writerow(["Nu:"])
                nus = ["%0.6f" % tmp['nu'] for tmp in iter_params]
                writer.writerow(nus)
            elif clf_name == 'IsolationForest':
                nus = ["%0.6f" % tmp['contamination'] for tmp in iter_params]
                writer.writerow(nus)
            else:
                writer.writerow(iter_params)
            writer.writerow("")
            # roc curve - save all precision and recall values

        # return parameter metrics
        if clf_name == 'OneClassSVM':
            return [
                nr_training_samples, nr_test_samples, nr_splits, np.median([tmp['nu'] for tmp in iter_params]),
                np.mean([tmp['nu'] for tmp in iter_params]), np.std([tmp['nu'] for tmp in iter_params]),
                np.mean(iter_precision), np.std(iter_precision), np.mean(iter_recall), np.std(iter_recall),
                np.mean(iter_f1_scores), np.std(iter_f1_scores),
                np.mean(iter_youden_indices), np.std(iter_youden_indices),
            ]
        elif clf_name == 'IsolationForest':
            return [
                nr_training_samples, nr_test_samples, nr_splits, np.median([tmp['contamination'] for tmp in iter_params]),
                np.mean([tmp['contamination'] for tmp in iter_params]), np.std([tmp['contamination'] for tmp in iter_params]),
                np.mean(iter_precision), np.std(iter_precision), np.mean(iter_recall), np.std(iter_recall),
                np.mean(iter_f1_scores), np.std(iter_f1_scores),
                np.mean(iter_youden_indices), np.std(iter_youden_indices),
            ]

# ================================= #
#              Main

if __name__ == '__main__':

    if False:
        params_svm = {'nu': np.arange(0.001, 0.1, 0.001), 'kernel': ['rbf']}
        clf = svm.OneClassSVM()
        tune_classifier(clf, params_svm, avg_cycles=3, nr_training_samples=20, nr_test_samples=120, combine_scenes=True)

    if False:
        params_if = {'contamination': np.arange(0.005, 0.1, 0.005)}
        clf = IsolationForest(random_state=np.random.RandomState(42))
        tune_classifier(clf, params_if)

    if False:
        params_abod = {'uncertainty_bandwidth': [0], 'threshold': np.arange(0.05, 0.5, 0.05)}
        clf = ABOD()
        tune_classifier(clf, params_abod, avg_cycles=2)

    # eval Isolation Forest
    if False:
        params_if = {'contamination': np.arange(0.005, 0.1, 0.005)}
        clf = IsolationForest(random_state=np.random.RandomState(42))
        eval = []
        # single scene
        eval.append(tune_classifier(clf, params_if, avg_cycles=5, nr_training_samples=10, nr_test_samples=640, combine_scenes=False, filename="IF_single-scene_train10_test640.csv"))
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=20, nr_test_samples=640, combine_scenes=False, filename="IF_single-scene_train20_test640.csv"))
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=40, nr_test_samples=640, combine_scenes=False, filename="IF_single-scene_train40_test640.csv"))
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=80, nr_test_samples=640, combine_scenes=False, filename="IF_single-scene_train80_test640.csv"))
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=160, nr_test_samples=640, combine_scenes=False, filename="IF_single-scene_train160_test640.csv"))
        # # multi scene
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=10, nr_test_samples=640, combine_scenes=True, filename="IF_multi-scene_train10_test640.csv"))
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=20, nr_test_samples=640, combine_scenes=True, filename="IF_multi-scene_train20_test640.csv"))
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=40, nr_test_samples=640, combine_scenes=True, filename="IF_multi-scene_train40_test640.csv"))
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=80, nr_test_samples=640, combine_scenes=True, filename="IF_multi-scene_train80_test640.csv"))
        # eval.append(tune_classifier(clf, params_if, avg_cycles=100, nr_training_samples=160, nr_test_samples=640, combine_scenes=True, filename="IF_multi-scene_train160_test640.csv"))
        # display results
        print "\n\n___________________________________________________\n\n"
        for e in eval:
            print e

    # eval OneClassSVM
    if True:
        params_svm = {'nu': np.arange(0.001, 0.1, 0.001), 'kernel': ['rbf']}
        clf = svm.OneClassSVM()
        eval = []
        # single scene
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=10, nr_test_samples=640, combine_scenes=False, filename="OCSVM_rbf_single-scene_train10_test640.csv"))
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=20, nr_test_samples=640, combine_scenes=False, filename="OCSVM_rbf_single-scene_train20_test640.csv"))
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=40, nr_test_samples=640, combine_scenes=False, filename="OCSVM_rbf_single-scene_train40_test640.csv"))
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=80, nr_test_samples=640, combine_scenes=False, filename="OCSVM_rbf_single-scene_train80_test640.csv"))
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=160, nr_test_samples=640, combine_scenes=False, filename="OCSVM_rbf_single-scene_train160_test640.csv"))
        # multi scene
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=10, nr_test_samples=640, combine_scenes=True, filename="OCSVM_rbf_multi-scene_train10_test640.csv"))
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=20, nr_test_samples=640, combine_scenes=True, filename="OCSVM_rbf_multi-scene_train20_test640.csv"))
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=40, nr_test_samples=640, combine_scenes=True, filename="OCSVM_rbf_multi-scene_train40_test640.csv"))
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=80, nr_test_samples=640, combine_scenes=True, filename="OCSVM_rbf_multi-scene_train80_test640.csv"))
        eval.append(tune_classifier(clf, params_svm, avg_cycles=100, nr_training_samples=160, nr_test_samples=640, combine_scenes=True, filename="OCSVM_rbf_multi-scene_train160_test640.csv"))
        # display results
        print "\n\n___________________________________________________\n\n"
        for e in eval:
            print e
