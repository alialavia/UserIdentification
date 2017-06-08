import numpy as np
import Queue
from sklearn.ensemble import IsolationForest
from uids.utils.Logger import Logger as log
# v2 models
from uids.v2.set_metrics import ABOD
from uids.v2.HardThreshold import SetSimilarityHardThreshold
from uids.data_models.MeanShiftCluster import MeanShiftCluster
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


# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')  # path to the model directory
ressourceDir = os.path.join(fileDir, '../..', 'ressource')	# path to the model directory


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

            # load images
            for i in indices:
                img_name = "{}/{}/{}".format(ressourceDir, img_folder_name, picture_names[i])
                print img_name
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


def test_external_cluster():

    ext_data = MeanShiftCluster()
    ext_data.update([[2, 1], [4, 65], [4, 3]])
    cls = SetSimilarityHardThreshold(metric='ABOD', threshold=0.7, cluster=ext_data)
    dec, scores = cls.predict([[2.2, 1], [4, 1]])
    print dec, scores


def test_queue_block():
    q = Queue.Queue(maxsize=0)
    q.put(1)
    q.put(2)

    while True:
        try:
            task = q.get(False)
        except Queue.Empty:
            # Handle empty queue here
            print "nooo"
            pass
            # check timeout trainings
        else:
            # Handle task here and call q.task_done()
            print "processing manual task"
            print task
            q.task_done()


def test_identification_controller():

    idc = IdentificationController({})

    is_save, samples = idc.accumulate_samples(2, np.array([1, 2, 3, 4, 5, 6]))
    print is_save, samples
    is_save, samples = idc.accumulate_samples(2, np.array([7, 8]))
    print is_save, samples
    # accumulation save now
    is_save, samples = idc.accumulate_samples(2, np.array([9, 10]))
    print is_save, samples
    # stack is deleted again
    is_save, samples = idc.accumulate_samples(2, np.array([11, 12]))
    print is_save, samples


def test_data_controller():
    dc = DataController()
    dc.add_samples(1337, np.array([1,2,3]))
    cluster = dc.get_class_cluster(1337)
    dc.add_samples(1337, np.array([4,5,6]))
    print cluster.data


def test_abod():

    emb2 = load_embeddings("matthias_test2.pkl")
    emb3 = load_embeddings("embeddings_christian_clean.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")


    test_with_ul = np.concatenate((emb2[1:3], emb3[0:10]))
    random.shuffle(test_with_ul)


    # print BaseMetaController.check_inter_sample_dist(test_with_ul)
    print BaseMetaController.check_inter_sample_dist(emb3)
    print BaseMetaController.check_inter_sample_dist(emb_lfw[0:1000])
    # print ABOD.get_set_score(emb3[0:4])
    # print ABOD.get_set_score(emb_lfw[0:4])

def test_normed_confidence():

    weights = np.array([2,6,9])
    predictions = [0,0,1]
    norm_f = 1.0/np.sum(weights)

    confidence = np.dot(predictions, np.transpose(norm_f * weights))
    print confidence


def test_arr_append():

    l = []
    l.append(2)
    l.append(4.2)
    l.append(np.array([3.4]))
    l.append(np.array([8.745]))
    print np.array(l).flatten()
    t = np.array([8.745, 9.234])
    print t
    print np.array(t).flatten()


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


def test_bias():
    values = np.array([1,2,3,4,5,1,2,6,1,4,78,1,4])
    # values = np.array([1.,2.,3.,4.,5.,1.,2.,6.,1.,4.,78.,1.,4.])
    weights = np.repeat(1.0, len(values))
    # unbiased estimator
    print np.var(values, ddof=1)
    # biased estimator
    print np.var(values)
    weighted_avg_and_var(values, weights)


def mix_crop(s1, s2, max_nr_samples=40):
    s1, s2 = shuffle(s1, s2, random_state=5)
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



def test_weighted_abod():
    # a = [-40, -19]
    #
    #
    #
    # a = np.clip(a, -30, 30)
    # print a
    #
    #
    # return

    abod_gen = WeightedABOD()
    grid = abod_gen.weight_gen

    # print abod_gen.weight_gen.euclidean_dist([30,0], [20,0])

    # training
    emb_train = load_embeddings("matthias_test2.pkl")
    pose_train = load_embeddings("matthias_test2_poses.pkl")

    # load test samples and poses
    emb1 = load_embeddings("matthias_pose_test3.pkl")
    pose1 = load_embeddings("matthias_pose_test3_poses.pkl")

    emb_ul1 = load_embeddings("christian_test1.pkl")
    pose_ul1 = load_embeddings("christian_test1_poses.pkl")
    emb_ul2 = load_embeddings("christian_test2.pkl")
    pose_ul2 = load_embeddings("christian_test2_poses.pkl")

    # mix sets and crop
    emb_train, pose_train = mix_crop(emb_train, pose_train, 60)
    emb1, pose1 = mix_crop(emb1, pose1, 40)
    emb_ul1, pose_ul1 = mix_crop(emb_ul1, pose_ul1, 40)
    # emb_ul2, pose_ul2 = mix_crop(emb_ul2, pose_ul2, 40)

    # drop roll
    pose_train = pose_train[:, 1:]
    pose1 = pose1[:, 1:]
    pose_ul1 = pose_ul1[:, 1:]

    # ---------------- DEBUG
    print "Calculating ABOD..."

    # print 173/162
    # print emb_train[173,0:3]
    # print emb_train[162,0:3]

    # print pose_train[0:5]


    # display nearest X images
    # display_image("matthias_test2", np.arange(0,5), img_folder_name="matthias_test2")

    if True:
        # select critial cases, compare
        abod_il = ABOD.get_score(emb1, emb_train)
        indices = np.arange(0, len(abod_il))
        sorted_indices = abod_il.argsort()

        # look at critical cases
        mask = abod_il < 0.3
        if np.count_nonzero(mask):

            print "____________________________________"
            print "Worst matches:"
            print "Poses: ", list(pose1[mask])

            display_image("matthias_pose_test3", indices[mask], img_folder_name="matthias_pose_test3")
            # print "Average distances: ", np.average(dist, axis=1)

            return



            abod_il_weighted, sample_weights = abod_gen.get_weighted_score(emb1[mask], pose1[mask], emb_train, pose_train)
            dist = pairwise_distances(emb1[mask], emb_train, metric='euclidean')
            dist = np.square(dist)



            # visualize worst matches
            display_image("matthias_pose_test3", indices[mask], img_folder_name="matthias_pose_test3")

            # display "best" corresponding match in db
            indices, weight = grid.best_subset(pose1[mask][1], pose_train)
            display_image("matthias_test2", sorted_indices[mask], img_folder_name="matthias_test2")




            print "weight: ", weight[0]
            dist = pairwise_distances(emb1[mask][1], emb_train[indices[0:1]], metric='euclidean')
            dist = np.square(dist)
            print "Average distances to subset: ", np.average(dist, axis=1)
            print "Critical cases ABOD: ", abod_il[mask]
            print "Critical cases ABOD_weighted: ", abod_il_weighted
            print "sample_weights: ", sample_weights








        #
        # print "ABOD_regular inlier: ", abod_il[sorted_indices[0:5]]
        #
        # abod_ul = ABOD.get_score(emb_ul1, emb_train)
        #
        #
        #
        #
        #
        #
        # sorted_indices = abod_ul.argsort()
        # print "ABOD_regular outlier: ", abod_ul[sorted_indices[-5:]]
        # print "ABOD_regular outlier: ", abod_ul[sorted_indices[0:5]]
        # print np.max(abod_ul)




    if False:
        i = 30
        print "Test pose: ", pose1[i]
        display_image("matthias_test2", np.arange(0,5), img_folder_name="matthias_test2")
        display_image("matthias_pose_test3", [i], img_folder_name="matthias_pose_test3")

        print "========= INLIER ==========="
        abod_val1, sample_weights = abod_gen.get_weighted_score(emb1[i:i+1], pose1[i:i+1], emb_train, pose_train)

        print "Test sample weight: ", sample_weights
        print "ABOD_weighted: ", abod_val1
        print "ABOD_regular: ", ABOD.get_score(emb1[i:i+1], emb_train)

        # ---------- distance comparison
        indices = grid.best_subset(pose1[i], pose_train)
        # calc dist
        dist_subset = pairwise_distances(emb1[i:i+1], emb_train[indices], metric='euclidean')[0]
        dist_subset = np.square(dist_subset)
        dist = pairwise_distances(emb1[i:i+1], emb_train, metric='euclidean')[0]
        dist = np.square(dist)
        print "Total distance mean: ", np.mean(dist)
        print "Subset distance mean: ", np.mean(dist_subset)


        print "========= OUTLIER ==========="
        abod_val1, sample_weights = abod_gen.get_weighted_score(emb_ul1[i:i+1], pose_ul1[i:i+1], emb_train, pose_train)
        print "Test pose: ", pose_ul1[i:i+1]
        display_image("christian_test1", [i], img_folder_name="christian_test1")
        print "Test sample weight: ", sample_weights
        print "ABOD_weighted: ", abod_val1
        print "ABOD_regular: ", ABOD.get_score(emb_ul1[i:i+1], emb_train)

        # select best matchin subset
        indices = grid.best_subset(pose_ul1[i], pose_train)
        abod_val1, sample_weights = abod_gen.get_weighted_score(emb_ul1[i:i + 1], pose_ul1[i:i + 1], emb_train[indices], pose_train[indices])
        print "ABOD_weighted_subset_select: ", abod_val1


        # ---------- subset subspace projection
        basis, mean = ExtractSubspace(emb_train[indices], 0.95)
        # project onto subset subspace
        train_reduced = ProjectOntoSubspace(emb_train, mean, basis)
        test_reduced = ProjectOntoSubspace(emb_ul1[i:i+1], mean, basis)
        # calc ABOD
        print "ABOD_regular on subset subspace: ", ABOD.get_score(test_reduced, train_reduced)


    if False:
        i = 8
        # print pose_train

        abod_val1, sample_weights = abod_gen.get_weighted_score(emb1[i:i+1], pose1[i:i+1], emb_train, pose_train)
        print "Test pose: ", pose1[i:i+1]
        print "Test sample weight: ", sample_weights
        print "ABOD_weighted: ", abod_val1

        # select best matchin subset
        indices = grid.best_subset(pose1[i], pose_train)

        # calc dist
        dist_subset = pairwise_distances(emb1[i:i+1], emb_train[indices], metric='euclidean')[0]
        dist_subset = np.square(dist_subset)
        dist = pairwise_distances(emb1[i:i+1], emb_train, metric='euclidean')[0]
        dist = np.square(dist)

        print "Total distance mean: ", np.mean(dist)
        print "Subset distance mean: ", np.mean(dist_subset)

        abod_val1, sample_weights = abod_gen.get_weighted_score(emb1[i:i + 1], pose1[i:i + 1], emb_train[indices], pose_train[indices])
        print "ABOD_weighted_subset_select: ", abod_val1
        print "ABOD_regular_subset_select: ", ABOD.get_score(emb1[i:i+1], emb_train[indices])
        abod_val1_regular = ABOD.get_score(emb1[i:i+1], emb_train)
        print "ABOD_regular: ", abod_val1_regular


    return



    print "-----------------------------------------"
    i = 2
    abod_val1, sample_weights = abod_gen.get_weighted_score(emb_ul1[i:i + 1], pose_ul1[i:i + 1], emb_train, pose_train)
    print "Test pose: ", pose_ul1[i:i + 1]
    print "Test sample weight: ", sample_weights
    print "ABOD_weighted: ", abod_val1
    abod_val1_regular = ABOD.get_score(emb_ul1[i:i + 1], emb_train)
    print "ABOD_regular: ", abod_val1_regular



    # ---------------- DEBUG
    return



    print "=========================="
    print "testing inliers (ABOD_weighted big)"
    print "__________________________"
    # classify with weights
    abod_val1, sample_weights = abod_gen.get_weighted_score(emb1, pose1, emb_train, pose_train)
    abod_sorted, sorted_indices = (list(t) for t in zip(*sorted(zip(abod_val1, np.arange(0,len(abod_val1))))))
    sep_sorted = np.array(abod_sorted)
    sorted_indices = np.array(sorted_indices)

    # worst indices:
    print "Worst ABOD values: ", sep_sorted[0:5]
    print "Worst pose: ", list(pose1[sorted_indices[0:5]])
    print "Worst weights: ", ["%0.2f" % i for i in list(sample_weights[sorted_indices[0:5]])]
    print "\n\n\n"

    # print "=========================="
    # print "testing inliers (ABOD_regular big)"
    # print "__________________________"
    # # classify without weights
    # abod_val1 = ABOD.get_score(emb1, emb_train)
    # abod_sorted, sorted_indices = (list(t) for t in zip(*sorted(zip(abod_val1, np.arange(0,len(abod_val1))), reverse=False)))
    # sep_sorted = np.array(abod_sorted)
    # sorted_indices = np.array(sorted_indices)
    # # worst indices:
    # print "Worst ABOD values: ", sep_sorted[0:5]
    # print "Worst pose: ", list(pose1[sorted_indices[0:5]])
    # print "\n\n\n"
    #
    # print "=========================="
    # print "testing outliers (ABOD_weighted small)"
    # print "__________________________"
    # # classify without weights
    # abod_val1, sample_weights = abod_gen.get_weighted_score(emb_ul1, pose_ul1, emb_train, pose_train)
    # abod_sorted, sorted_indices = (list(t) for t in zip(*sorted(zip(abod_val1, np.arange(0,len(abod_val1))), reverse=True)))
    # sep_sorted = np.array(abod_sorted)
    # sorted_indices = np.array(sorted_indices)
    # # worst indices:
    # print "Worst ABOD values: ", sep_sorted[0:5]
    # print "Worst pose: ", list(pose1[sorted_indices[0:5]])
    # print "Worst weights: ", ["%0.2f" % i for i in list(sample_weights[sorted_indices[0:5]])]
    # print "\n\n\n"
    #
    # print "=========================="
    # print "testing outliers (ABOD_regular small)"
    # print "__________________________"
    # # classify without weights
    # abod_val1 = ABOD.get_score(emb_ul1, emb_train)
    # abod_sorted, sorted_indices = (list(t) for t in zip(*sorted(zip(abod_val1, np.arange(0,len(abod_val1))), reverse=True)))
    # sep_sorted = np.array(abod_sorted)
    # sorted_indices = np.array(sorted_indices)
    # # worst indices:
    # print "Worst ABOD values: ", sep_sorted[0:5]
    # print "Worst pose: ", list(pose1[sorted_indices[0:5]])
    # print "\n\n\n"


if __name__ == '__main__':
    test_var()


