import numpy as np
import Queue
from sklearn.ensemble import IsolationForest
from uids.utils.Logger import Logger as log
# v2 models
from uids.v2.set_metrics import ABOD
from uids.v2.HardThreshold import SetSimilarityHardThreshold
from uids.data_models.StandardCluster import StandardCluster
from uids.v2.MultiClassClassifierBase import MultiClassClassifierBase
from uids.v2.DataController import DataController
from uids.v2.ClassifierController import IdentificationController, UpdateController, BaseMetaController
from uids.v2.set_metrics import *
import os
import random
import pickle
from sklearn.utils import shuffle
from uids.utils.DataAnalysis import *
from scipy import misc
from sklearn import metrics
import time
import csv
from numpy.random import RandomState


# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embeddings_eval')  # path to the model directory
ressourceDir = os.path.join(fileDir, '../..', 'ressource')	# path to the model directory

current_milli_time = lambda: int(round(time.time() * 1000))

def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def test_metrics():
    cls = SetSimilarityHardThreshold(metric='ABOD', threshold=0.7)
    cls.partial_fit([[2, 1], [4, 65], [4, 3]])
    dec, scores = cls.predict([[2.2, 1], [4, 1]])
    print dec, scores


def clean_duplicates(s1, s2):
    assert len(s1) == len(s2)

    x = np.random.rand(s1.shape[1])
    y = s1.dot(x)
    unique, index = np.unique(y, return_index=True)

    # print len(s1)
    # print len(s1[index])
    # print len(np.vstack({tuple(row) for row in s1}))

    if len(s1[index]) != len(s1):
        log.severe("Duplicate items in embeddings s1! Removing duplicates...")
        s1 = s1[index]
        s2 = s2[index]
        # raise ValueError

    return s1, s2


def mix_crop(s1, s2, max_nr_samples=40, random_state=None):

    assert len(s1) == len(s2)

    if random_state is not None:
        prng = RandomState(random_state)
        indices_mixed = prng.permutation(np.arange(0, len(s1)))
        s1 = s1[indices_mixed]
        s2 = s2[indices_mixed]

    max_nr_samples = max_nr_samples if max_nr_samples < len(s1) else len(s1)
    s1 = s1[0:max_nr_samples]
    s2 = s2[0:max_nr_samples]
    return s1, s2


def eval_wabod(
        ref_emb="matthias_test.pkl",
        ref_poses="matthias_test_poses.pkl",
        test_inlier_emb="matthias1.pkl",
        test_inlier_poses="matthias1_poses.pkl",
        test_outlier_emb="christian_test2.pkl",
        test_outlier_poses="christian_test2_poses.pkl"
):

    # mix sets and crop
    training_set_sizes = [5, 10, 20, 30, 40, 50, 60]
    # training_set_sizes = [10]
    nr_test_images = 100
    nr_random_iters = 10

    # training
    emb_train = load_embeddings(ref_emb)
    pose_train = load_embeddings(ref_poses)

    # load test samples and poses
    emb1 = load_embeddings(test_inlier_emb)
    pose1 = load_embeddings(test_inlier_poses)

    emb2 = load_embeddings(test_outlier_emb)
    pose2 = load_embeddings(test_outlier_poses)

    # clean duplicates
    emb_train, pose_train = clean_duplicates(emb_train, pose_train)
    emb1, pose1 = clean_duplicates(emb1, pose1)
    emb2, pose2 = clean_duplicates(emb2, pose2)

    # drop roll
    pose_train = pose_train[:, 1:]
    pose1 = pose1[:, 1:]
    pose2 = pose2[:, 1:]

    if len(emb1) < nr_test_images:
        print "Not enough images"
        raise ValueError
    if len(emb2) < nr_test_images:
        print "Not enough images"
        raise ValueError

    emb1, pose1 = mix_crop(emb1, pose1, nr_test_images, random_state=None)
    emb2, pose2 = mix_crop(emb2, pose2, nr_test_images, random_state=None)

    emb_test = np.concatenate((emb1, emb2))
    pose_test = np.concatenate((pose1, pose2))

    # ------------ start evaluation

    abod_gen = WeightedABOD(variant=2)
    grid = abod_gen.weight_gen

    auc_avg = {
        'abod': [],
        'wabod1': [],
        'wabod2': [],
        'wabod3': [],
        'wabod4': [],
        'wabod5': []
    }

    pred_time_avg = {
        'abod': [],
        'wabod1': [],
        'wabod2': [],
        'wabod3': [],
        'wabod4': [],
        'wabod5': []
    }

    for t_size in training_set_sizes:

        auc_values = {
            'abod': [],
            'wabod1': [],
            'wabod2': [],
            'wabod3': [],
            'wabod4': [],
            'wabod5': []
        }
        pred_time = {
            'abod': [],
            'wabod1': [],
            'wabod2': [],
            'wabod3': [],
            'wabod4': [],
            'wabod5': []
        }

        sys.stdout.write("Training with size {} ".format(t_size))
        start = time.time()

        for i in range(0, nr_random_iters):
            if i == 1:
                est_time = (time.time()-start) * nr_random_iters
                if est_time > 60:
                    est_time = est_time/60.0
                    sys.stdout.write(" | Estimated time: {:.2f} min, Iteration: ".format(est_time))
                else:
                    sys.stdout.write(" | Estimated time: {:.2f} sec, Iteration: ".format(est_time))
            if i > 0:
                sys.stdout.write("{}, ".format(i+1))

            # select training subset
            emb_train_subset, pose_train_subset = mix_crop(emb_train, pose_train, t_size, random_state=(i+1))
            true_labels = np.concatenate((np.repeat(1, len(emb1)), np.repeat(0, len(emb2))))

            # calc abod variants
            start = current_milli_time()
            abod_score = ABOD.get_score(emb_test, emb_train_subset)
            pred_time['abod'].append(current_milli_time() - start)
            abod_gen.set_params(**{"variant": 1})
            start = current_milli_time()
            wabod1_score, weights = abod_gen.get_weighted_score(emb_test, pose_test, emb_train_subset, pose_train_subset)
            pred_time['wabod1'].append(current_milli_time() - start)
            abod_gen.set_params(**{"variant": 2})
            start = current_milli_time()
            wabod2_score, weights = abod_gen.get_weighted_score(emb_test, pose_test, emb_train_subset, pose_train_subset)
            pred_time['wabod2'].append(current_milli_time() - start)
            abod_gen.set_params(**{"variant": 3})
            start = current_milli_time()
            wabod3_score, weights = abod_gen.get_weighted_score(emb_test, pose_test, emb_train_subset, pose_train_subset)
            pred_time['wabod3'].append(current_milli_time() - start)
            abod_gen.set_params(**{"variant": 4})
            start = current_milli_time()
            wabod4_score, weights = abod_gen.get_weighted_score(emb_test, pose_test, emb_train_subset, pose_train_subset)
            pred_time['wabod4'].append(current_milli_time() - start)
            abod_gen.set_params(**{"variant": 5})
            start = current_milli_time()
            wabod5_score, weights = abod_gen.get_weighted_score(emb_test, pose_test, emb_train_subset, pose_train_subset)
            pred_time['wabod5'].append(current_milli_time() - start)

            # calc area under curve
            auc_val = roc_auc_score(true_labels, abod_score, sample_weight=None)
            auc_values['abod'].append(auc_val)
            auc_val = roc_auc_score(true_labels, wabod1_score, sample_weight=None)
            auc_values['wabod1'].append(auc_val)
            auc_val = roc_auc_score(true_labels, wabod2_score, sample_weight=None)
            auc_values['wabod2'].append(auc_val)
            auc_val = roc_auc_score(true_labels, wabod3_score, sample_weight=None)
            auc_values['wabod3'].append(auc_val)
            auc_val = roc_auc_score(true_labels, wabod4_score, sample_weight=None)
            auc_values['wabod4'].append(auc_val)
            auc_val = roc_auc_score(true_labels, wabod5_score, sample_weight=None)
            auc_values['wabod5'].append(auc_val)

        print "\n"
        # auc_avg['abod'].append(np.average(auc_values['abod']))
        # auc_avg['wabod1'].append(np.average(auc_values['wabod1']))
        # auc_avg['wabod2'].append(np.average(auc_values['wabod2']))
        # auc_avg['wabod3'].append(np.average(auc_values['wabod3']))
        # auc_avg['wabod4'].append(np.average(auc_values['wabod4']))
        # auc_avg['wabod5'].append(np.average(auc_values['wabod5']))

        for k in auc_values.keys():
            auc_avg[k].append(np.average(auc_values[k]))

        for k in pred_time.keys():
            pred_time_avg[k].append(np.average(pred_time[k]))

    # ----------------------------------- save results to file

    # save results to file
    filename = "WABOD_variant_performance_{}.csv".format(int(time.time()))
    print "Saving file to: {}".format(filename)
    with open(filename, 'wb') as csvfile:
        # write configuration of best results over multiple random tests
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(["AUC - Area Under Curve Evaluation"])
        writer.writerow(["Nr Iterations", nr_random_iters])
        writer.writerow(["Test Set size (total)", nr_test_images * 2])
        writer.writerow("")
        writer.writerow(["Datasets"])
        writer.writerow(["Training", "Test Inliers", "Test Outliers"])
        writer.writerow([ref_emb, test_inlier_emb, test_outlier_emb])
        writer.writerow("")
        writer.writerow(["Unweighted Average"])  # iterate over
        writer.writerow(["Nr. Training Samples"] + training_set_sizes)
        for type in auc_avg.keys():
            writer.writerow([type] + auc_avg[type])
        writer.writerow("")
        writer.writerow(["Prediction time"])
        writer.writerow(["Nr. Training Samples"] + training_set_sizes)
        for type in pred_time_avg.keys():
            writer.writerow([type] + pred_time_avg[type])


if __name__ == '__main__':

    # evaluation 1
    eval_wabod(
        ref_emb="matthias_test2.pkl",
        ref_poses="matthias_test2_poses.pkl",
        test_inlier_emb="matthias1.pkl",
        test_inlier_poses="matthias1_poses.pkl",
        test_outlier_emb="daniel_scene2_clean.pkl",
        test_outlier_poses="daniel_scene2_clean_poses.pkl"
    )