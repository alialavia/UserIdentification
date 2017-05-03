import numpy as np
from uids.utils.Logger import Logger as log
# v2 models
from uids.v2.set_metrics import ABOD
from uids.v2.HardThreshold import SetSimilarityHardThreshold
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


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def plot_roc(y_true, scores, pos_label=1, sample_weight=None):

    # remember:

    # fpr, tpr, thresholds = roc_curve(y, scores)
    # roc_auc = auc(fpr, tpr)
    # pl.plot(fpr, tpr)
    #
    # precision, recall, thresholds = precision_recall_curve(y, scores)
    # pr_auc = auc(recall, precision)
    # pl.plot(recall, precision)


    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores, sample_weight=sample_weight, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.5f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()

def display_image(embedding_name, indices, img_folder_name=""):

    filename = "{}/{}_image_names.pkl".format(modelDir, embedding_name)

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            picture_names = pickle.load(f)
            f.close()
            if img_folder_name == "":
                img_folder_name = embedding_name

            size = 250
            images = []
            widths = []
            heights = []


            if(os.path.isdir("{}/{}".format(ressourceDir, img_folder_name+"_prealigned"))):
                img_folder_name += "_prealigned"

            # load images
            for i in indices:
                img_name = "{}/{}/{}".format(ressourceDir, img_folder_name, picture_names[i])
                # print img_name
                image = misc.imread(img_name)
                height, width, dims = image.shape
                image = misc.imresize(image, (size, size))
                images.append(image)
                widths.append(width)
                heights.append(height)

            new_im = np.hstack(images)
            # print new_im.shape
            plt.imshow(new_im, aspect="auto")
            plt.show()

            # misc.pilutil.imshow(image)
    else:
        print "Can not load image names in: {}".format(filename)


def weighted_avg_and_var(values, weights, normalized_weights=False):
    """
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # print "avg: {}, weighted: {}".format(np.average(values), average)
    variance_biased = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    # print "variance biased: ", variance_biased
    V1 = np.sum(weights)
    # V1_sqr = V1**2
    V2 = np.sum(weights**2)
    variance_unbiased = variance_biased/(1.-(V2/(V1**2)))
    # variance_unbiased = (V1_sqr/(V1_sqr-V2))*variance_biased
    # print "variance unbiased: ", variance_unbiased
    return (average, variance_unbiased)


def mix_crop(s1, s2, max_nr_samples=40, random_state=None):
    if random_state is not None:
        # prng = RandomState(random_state)
        # s1 = prng.permutation(s1)
        # s2 = prng.permutation(s2)
        s1, s2 = shuffle(s1, s2, random_state=random_state)
    max_nr_samples = max_nr_samples if max_nr_samples < len(s1) else len(s1)
    s1 = s1[0:max_nr_samples]
    s2 = s2[0:max_nr_samples]
    return s1, s2


def weighted_var(values, weights):
    """
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # print "avg: {}, weighted: {}".format(np.average(values), average)
    variance_biased = np.average((values-average)**2)  # Fast and numerically precise
    # print "variance biased: ", variance_biased
    return variance_biased


def test_var():
    samples = [1,1.1,1,1,1.2,5,8,2,3]
    weights = [0.1,0.9,0.9,0.2,0.15, 0.8, 0.7, 0.9,0.7]
    # expected: high
    print np.var(samples)
    print weighted_var(samples, weights)


def profile(emb_train, pose_train, emb_test, pose_test, name_train="matthias1", name_test="matthias2", is_inlier=True):

    visualize = False

    abod_gen = WeightedABOD()
    grid = abod_gen.weight_gen

    # select critial cases, compare
    abod_il = ABOD.get_score(emb_test, emb_train)
    indices = np.arange(0, len(abod_il))

    # look at critical cases

    log.print_clr("EVALUATING {}".format("Inlier" if is_inlier else "Outlier"), 'red', 'black')

    if is_inlier:
        mask = abod_il < 0.3
        sorted_indices = abod_il.argsort()
    else:
        mask = abod_il > 0.3
        sorted_indices = abod_il.argsort()[::-1]

    nr_failures = np.count_nonzero(mask)

    if np.count_nonzero(mask) == 0:
        print "no misdetections..."
    else:
        abod_il_weighted, sample_weights = abod_gen.get_weighted_score(emb_test[mask], pose_test[mask], emb_train, pose_train)

        print "____________________________________"
        print "Worst matches:"
        print "Too many misdetections...  {}".format(np.count_nonzero(mask))
        if np.count_nonzero(mask) > 5:
            pass
        else:
            print "Poses: ", list(pose_test[mask])
        print "Regular ABOD: ", abod_il[mask]
        print "ABOD_weighted: ", abod_il_weighted

        worst_pose = pose_test[sorted_indices[0]]
        worst_emb = emb_test[sorted_indices[0]]

        # show best matches in db for worst image
        nearest_indices, pose_separation = grid.best_subset(worst_pose, pose_train)
        print "Most compatible poses: {} (test)... {} (ref)".format(worst_pose, pose_train[nearest_indices[0]])
        print "Pose separation: ", pose_separation[0]
        # ------------------------ calc euclidean distance to nearest matches

        nearest_dist = pairwise_distances(worst_emb.reshape(1, -1), emb_train[nearest_indices][0], metric='euclidean')
        nearest_dist = np.square(nearest_dist)
        print "L2_squared to nearest in db: ", nearest_dist

        dist = pairwise_distances(worst_emb.reshape(1, -1), emb_train, metric='euclidean')[0]
        dist = np.square(dist)
        avg_dist = np.mean(dist)
        print "Average distance to db: ", avg_dist

        weights = [abod_gen.weight_gen.get_weight(worst_emb, p) for p in pose_train]
        wavg_dist = np.average(dist, weights=weights)
        print "Weighted average distance to db: ", wavg_dist

        # --------------- evaluate

        if is_inlier:
            # if abod_il_weighted[sorted_indices[0]] > abod_il[sorted_indices[0]]:
            #     log.print_clr("WABOD better!")

            if wavg_dist < avg_dist:
                log.print_clr("Wheighted avg dist < avg dist", 'red', 'white')
        else:
            # if abod_il_weighted[sorted_indices[0]] < abod_il[sorted_indices[0]]:
            #     log.print_clr("WABOD better!")

            if wavg_dist > avg_dist:
                log.print_clr("Wheighted avg dist > avg dist", 'red', 'white')

        # visualize critical case
        if visualize and np.count_nonzero(mask)<7:
            plt.figure("Worst test images [left-to-right]")
            display_image(name_test, sorted_indices[0:nr_failures], img_folder_name=name_test)
            plt.figure("Best match for worst image in db")
            display_image(name_train, nearest_indices[0:1], img_folder_name=name_train)


def test_weighted_abod():

    abod_gen = WeightedABOD()
    grid = abod_gen.weight_gen

    print grid.get_weight([24,-18], [13,36])

    # print abod_gen.weight_gen.euclidean_dist_squared([30,0], [20,0])

    # training
    emb_train = load_embeddings("matthias1.pkl")
    pose_train = load_embeddings("matthias1_poses.pkl")

    # load test samples and poses
    emb1 = load_embeddings("matthias2.pkl")
    pose1 = load_embeddings("matthias2_poses.pkl")

    emb2 = load_embeddings("christian_test1.pkl")
    pose2 = load_embeddings("christian_test1_poses.pkl")

    # drop roll
    pose_train = pose_train[:, 1:]
    pose1 = pose1[:, 1:]

    # mix sets and crop
    emb_train, pose_train = mix_crop(emb_train, pose_train, 10)
    emb1, pose1 = mix_crop(emb1, pose1, 200)


    # ---------------- CHECK
    # display_image("matthias1", [0], img_folder_name="matthias1")
    # print

    # ---------------- DEBUG
    print "Calculating ABOD..."

    # display nearest X images
    # display_image("matthias1", [2], img_folder_name="matthias1")

    profile(emb_train, pose_train, emb1, pose1, name_train="matthias1", name_test="matthias2", is_inlier=True)
    # profile(emb_train, pose_train, emb2, pose2, name_train="matthias1", name_test="christian_test1", is_inlier=False)


    #
    # # visualize worst matches
    # display_image("matthias_pose_test3", indices[mask], img_folder_name="matthias_pose_test3")
    #
    # # display "best" corresponding match in db
    # indices, weight = grid.best_subset(pose1[mask][1], pose_train)
    # display_image("matthias_test2", sorted_indices[mask], img_folder_name="matthias_test2")
    #
    #


    # print "weight: ", weight[0]
    # dist = pairwise_distances(emb1[mask][1], emb_train[indices[0:1]], metric='euclidean')
    # dist = np.square(dist)
    # print "Average distances to subset: ", np.average(dist, axis=1)
    # print "Critical cases ABOD: ", abod_il[mask]
    # print "Critical cases ABOD_weighted: ", abod_il_weighted
    # print "sample_weights: ", sample_weights
    #
    #


def generate_roc():
    abod_gen = WeightedABOD()
    grid = abod_gen.weight_gen

    # training
    emb_train = load_embeddings("matthias1.pkl")
    pose_train = load_embeddings("matthias1_poses.pkl")

    # load test samples and poses
    emb1 = load_embeddings("matthias2.pkl")
    pose1 = load_embeddings("matthias2_poses.pkl")

    emb2 = load_embeddings("christian_test2.pkl")
    pose2 = load_embeddings("christian_test2_poses.pkl")

    # drop roll
    pose_train = pose_train[:, 1:]
    pose1 = pose1[:, 1:]
    pose2 = pose2[:, 1:]

    # mix sets and crop
    emb_train, pose_train = mix_crop(emb_train, pose_train, 5)
    emb1, pose1 = mix_crop(emb1, pose1, 200)
    emb2, pose2 = mix_crop(emb2, pose2, 200)

    # calc regular abod
    abod_il = ABOD.get_score(emb1, emb_train)
    abod_ol = ABOD.get_score(emb2, emb_train)
    abod_scores = np.concatenate((abod_il, abod_ol))
    labels_regular = np.concatenate((np.repeat(1,len(abod_il)), np.repeat(-1,len(abod_ol))))

    print "Calculating weighted ABOD..."

    # calc weighted abod
    abod_il_weighted, sample_weights_il = abod_gen.get_weighted_score(emb1, pose1, emb_train, pose_train)
    abod_ol_weighted, sample_weights_ul = abod_gen.get_weighted_score(emb2, pose2, emb_train, pose_train)
    abod_scores_weighted = np.concatenate((abod_il_weighted, abod_ol_weighted))
    weights = np.concatenate((sample_weights_il, sample_weights_ul))
    print weights

    print "ABOD calculation complete. Plotting..."
    plt.figure('Regular ABOD')
    plot_roc(labels_regular, abod_scores, pos_label=1)
    plt.figure('Weighted ABOD')
    plot_roc(labels_regular, abod_scores_weighted, pos_label=1)
    plt.figure('Weighted ABOD with weights')
    plot_roc(labels_regular, abod_scores_weighted, pos_label=1, sample_weight=weights)


def eval_wabod():

    # mix sets and crop
    training_set_sizes = [20]
    nr_test_images = 100
    nr_random_iters = 2

    # training
    emb_train = load_embeddings("matthias_test.pkl")
    pose_train = load_embeddings("matthias_test_poses.pkl")

    # load test samples and poses
    emb1 = load_embeddings("matthias1.pkl")
    pose1 = load_embeddings("matthias1_poses.pkl")

    emb2 = load_embeddings("christian_test2.pkl")
    pose2 = load_embeddings("christian_test2_poses.pkl")

    # drop roll
    pose_train = pose_train[:, 1:]
    pose1 = pose1[:, 1:]
    pose2 = pose2[:, 1:]

    if len(emb1) < nr_test_images:
        print "Not enough images"
        raise ValueError
    if len(emb1) < nr_test_images:
        print "Not enough images"
        raise ValueError

    emb1, pose1 = mix_crop(emb1, pose1, nr_test_images, random_state=None)
    emb2, pose2 = mix_crop(emb2, pose2, nr_test_images, random_state=None)

    # ------------ start evaluation

    abod_gen = WeightedABOD()
    grid = abod_gen.weight_gen

    aoc_avg = {
        'abod': [],
        'wabod': [],
        'abod_w': [],
        'wabod_w': []
    }

    aoc_wavg = {
        'abod': [],
        'wabod': [],
        'abod_w': [],
        'wabod_w': []
    }

    prediction_time_avg = {
        'abod': [],
        'wabod': []
    }

    for t_size in training_set_sizes:

        roc_auc_comparison = {
            'abod': [],
            'abod_w': [],
            'wabod': [],
            'wabod_w': []
        }
        prediction_time = {
            'abod': [],
            'wabod': []
        }

        iter_weights = []

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
            emb_train_subset, pose_train_subset = mix_crop(emb_train, pose_train, t_size, random_state=i+1)
            true_labels = np.concatenate((np.repeat(1,len(emb1)), np.repeat(0,len(emb2))))

            # calc regular abod
            start = time.time()
            abod_il = ABOD.get_score(emb1, emb_train_subset)
            abod_ol = ABOD.get_score(emb2, emb_train_subset)
            prediction_time['abod'].append(time.time() - start)
            abod_scores = np.concatenate((abod_il, abod_ol))

            # calc weighted abod
            start = time.time()
            abod_il_weighted, sample_weights_il = abod_gen.get_weighted_score(emb1, pose1, emb_train_subset, pose_train_subset)
            abod_ol_weighted, sample_weights_ul = abod_gen.get_weighted_score(emb2, pose2, emb_train_subset, pose_train_subset)
            prediction_time['wabod'].append(time.time() - start)
            abod_scores_weighted = np.concatenate((abod_il_weighted, abod_ol_weighted))
            weights = np.concatenate((sample_weights_il, sample_weights_ul))

            # get area under curve
            # fpr, tpr, thresholds = metrics.roc_curve(true_labels, abod_scores, sample_weight=None, pos_label=1)
            # roc_auc_comparison['abod'].append(auc(fpr, tpr))
            # fpr, tpr, thresholds = metrics.roc_curve(true_labels, abod_scores_weighted, sample_weight=None, pos_label=1)
            # roc_auc_comparison['wabod'].append(auc(fpr, tpr))
            # fpr, tpr, thresholds = metrics.roc_curve(true_labels, abod_scores_weighted, sample_weight=weights, pos_label=1)
            # roc_auc_comparison['wwabod'].append(auc(fpr, tpr))

            # iteration metrics
            auc_val = roc_auc_score(true_labels, abod_scores, sample_weight=None)
            roc_auc_comparison['abod'].append(auc_val)
            auc_val = roc_auc_score(true_labels, abod_scores, sample_weight=weights)
            roc_auc_comparison['abod_w'].append(auc_val)
            auc_val = roc_auc_score(true_labels, abod_scores_weighted, sample_weight=None)
            roc_auc_comparison['wabod'].append(auc_val)
            auc_val = roc_auc_score(true_labels, abod_scores_weighted, sample_weight=weights)
            roc_auc_comparison['wabod_w'].append(auc_val)

            # iteration weight - average of sample weights
            iter_weights.append(np.mean(abod_scores_weighted))

        print "\n"

        # regular mean
        aoc_avg['abod'].append(np.mean(roc_auc_comparison['abod']))
        aoc_avg['wabod'].append(np.mean(roc_auc_comparison['wabod']))

        # regular mean of weighted metrics
        aoc_avg['abod_w'].append(np.mean(roc_auc_comparison['abod_w']))
        aoc_avg['wabod_w'].append(np.mean(roc_auc_comparison['wabod_w']))

        # weighted mean of regular metrics
        aoc_wavg['abod'].append(np.average(roc_auc_comparison['abod'], weights=iter_weights))
        aoc_wavg['wabod'].append(np.average(roc_auc_comparison['wabod'], weights=iter_weights))

        # weighted mean of weighted metrics
        aoc_wavg['abod_w'].append(np.average(roc_auc_comparison['abod_w'], weights=iter_weights))
        aoc_wavg['wabod_w'].append(np.average(roc_auc_comparison['wabod_w'], weights=iter_weights))

        prediction_time_avg['abod'].append(np.mean(prediction_time['abod']))
        prediction_time_avg['wabod'].append(np.mean(prediction_time['wabod']))

    # save results to file
    filename = 'ABOD_variant_performance.csv'
    print "Saving file to: {}".format(filename)
    with open(filename, 'wb') as csvfile:
        # write configuration of best results over multiple random tests
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(["AUC - Area Under Curve Evaluation"])
        writer.writerow(["Nr Iterations", nr_random_iters])
        writer.writerow(["Test Set size (total)", nr_test_images*2])
        writer.writerow("")
        writer.writerow(["Unweighted Average"])        # iterate over
        writer.writerow(["Nr. Training Samples"] + training_set_sizes)
        for type in aoc_avg.keys():
            writer.writerow([type] + aoc_avg[type])

        writer.writerow("")
        writer.writerow(["Weighted Average"])        # iterate over
        writer.writerow(["Nr. Training Samples"] + training_set_sizes)
        for type in aoc_wavg.keys():
            writer.writerow([type] + aoc_wavg[type])

        writer.writerow("")
        writer.writerow(["Prediction Time"])
        writer.writerow(["Nr. Training Samples"] + training_set_sizes)
        for type in prediction_time_avg.keys():
            writer.writerow([type] + prediction_time_avg[type])


    # # ------------ plot results
    # for type in roc_auc_comparison_avg.keys():
    #     plt.figure("Average AOC - {}".format(type))
    #     print "Plotting type ", type
    #     print roc_auc_comparison_avg[type]
    #     plt.plot(training_set_sizes, roc_auc_comparison_avg[type])
    #     plt.ylim([-0.1, 1.2])
    #     plt.xlabel('Training Set Size')
    #     plt.ylabel('Average AOC')
    #     plt.title('ROC curve')
    #     plt.show()
    #
    # plt.show()


if __name__ == '__main__':

    # y = np.array([1, 1, 2, 2])
    # scores = np.array([0.1, 0.4, 0.35, 0.8])
    # plot_roc(y, scores, pos_label=1)

    eval_wabod()



