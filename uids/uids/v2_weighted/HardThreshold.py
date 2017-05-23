from uids.utils.DataAnalysis import *
from uids.data_models.MeanShiftCluster import MeanShiftCluster
from uids.data_models.ClusterBase import ClusterBase
from sklearn.metrics.pairwise import *
import time
from uids.utils.Logger import Logger as log
from abc import abstractmethod


class SetSimilarityThresholdBase:
    """
    SetSimilarityThreshold calculates a per-sample outlier/similarity score which is thresholded for classification
    """

    __verbose = False
    data_cluster = None

    # hashed result buffer
    decision_fn_buffer = {}
    matching_conf_buffer = {}

    def __init__(self, cluster, metric='ABOD'):
        self.data_cluster = cluster
        self.metric = metric
        self.cluster_timestamp = time.time()

    def partial_fit(self, samples):
        # invalid buffered decision function
        self.decision_fn_buffer = {}
        self.matching_conf_buffer = {}

    def get_hash(self, arr):
        arr.flags.writeable = False
        h = hash(arr.data)
        arr.flags.writeable = True
        return h

    def decision_function(self, samples, samples_poses, nr_compaired_samples):
        """
        Distance of the samples X to the target class distribution
        :param samples:
        :return:
        """

        # calc hashes
        hashed = [self.get_hash(s) for s in samples]

        # check intersections and use buffered results
        if self.decision_fn_buffer:
            # ind_samples = dict((k, i) for i, k in enumerate(hashed))
            intersec_hashes = list(set(self.decision_fn_buffer.keys()) & set(hashed))

            similarity_scores = []
            matching_confidence = []

            for i, h in enumerate(hashed):

                if h in intersec_hashes:
                    similarity_scores.append(self.decision_fn_buffer[h])
                    matching_confidence.append(self.matching_conf_buffer[h])
                else:
                    abod_scores, confidence_scores = self.data_cluster.sample_set_similarity_scores(
                        np.array([samples[i]]), samples_poses, self.metric, nr_ref_samples=nr_compaired_samples
                    )

                    similarity_scores.append(abod_scores[0])
                    matching_confidence.append(confidence_scores[0])

                    # add to buffer
                    self.decision_fn_buffer[h] = abod_scores[0]
                    self.matching_conf_buffer[h] = confidence_scores[0]

            # print "sim scores1: ", similarity_scores
        else:
            similarity_scores, matching_confidence = self.data_cluster.sample_set_similarity_scores(
                samples, samples_poses, self.metric, nr_ref_samples=nr_compaired_samples
            )

            # print "sim scores2: ", similarity_scores

            # add to buffer
            for i, h in enumerate(hashed):
                self.decision_fn_buffer[h] = similarity_scores[i]
                self.matching_conf_buffer[h] = matching_confidence[i]

        similarity_scores = np.array(similarity_scores).flatten()
        matching_confidence = np.array(matching_confidence).flatten()

        return similarity_scores, matching_confidence

    @abstractmethod
    def predict(self, samples, samples_poses):
        """
        Specifies how to update self.data with incomming samples
        """
        raise NotImplementedError("Implement Cluster Update.")


class SetSimilarityHardThreshold(SetSimilarityThresholdBase):

    # hard decision threshold
    __thresh = None
    metric = None
    nr_compaired_samples = 0
    recheck_L2_distance = True

    def __init__(self, threshold=0.3, cluster=None, metric='ABOD', nr_compaired_samples=40, recheck_l2=False):
        SetSimilarityThresholdBase.__init__(self, cluster=cluster)
        self.__thresh = threshold
        self.metric = metric
        self.nr_compaired_samples = nr_compaired_samples
        self.recheck_L2_distance = recheck_l2

    def predict(self, samples, samples_poses):

        # get similarity scores
        cluster_type = self.data_cluster.__class__.__name__
        if cluster_type == 'MeanShiftPoseCluster':
            similarity_scores, matching_confidence = self.decision_function(
                samples, samples_poses, nr_compaired_samples=self.nr_compaired_samples
            )
        else:
            log.severe("Prediction for cluster type '{}' is not implemented yet!".format(cluster_type))
            raise NotImplementedError("Implement threshold prediction for specific cluster type.")

        print "==== {}: ".format(self.metric), ["%0.3f" % i for i in similarity_scores]
        l2_dist = self.data_cluster.class_mean_dist(samples, metric='euclidean')
        print "==== L2: ", ["%0.3f" % i for i in l2_dist]
        print "==== Matching conf: ", ["%0.1f" % i for i in matching_confidence]

        if self.metric == 'ABOD':
            positive = similarity_scores > self.__thresh

            # only apply on 50% max of samples
            if self.recheck_L2_distance and np.count_nonzero(positive) >= int(len(positive)/2.):
                m1 = similarity_scores >= 0.16
                m2 = l2_dist < 0.6
                print ".... Rechecking L2 distance, detections: ", m1 & m2
                positive[m1 & m2] = True
        else:
            positive = similarity_scores < self.__thresh
        return np.array([1 if v else -1 for v in positive]), np.array(matching_confidence)

