from uids.utils.DataAnalysis import *
from uids.data_models.StandardCluster import StandardCluster
from uids.data_models.ClusterBase import ClusterBase
from sklearn.metrics.pairwise import *
import time

class SetSimilarityThresholdBase:
    """
    SetSimilarityThreshold calculates a per-sample outlier/similarity score which is thresholded for classification
    """

    __verbose = False
    __external_cluster = True   # has an external data model
    data_cluster = None

    # hashed result buffer
    decision_fn_buffer = {}

    def __init__(self, cluster=None, metric='ABOD'):

        if cluster is None:
            print "No data cluster linked. Using new StandardCluster."
            self.data_cluster = StandardCluster()
            self.__external_cluster = False
        else:
            self.data_cluster = cluster

        self.metric = metric
        self.cluster_timestamp = time.time()

    def partial_fit(self, samples):
        if self.__external_cluster:
            # DONT UPDATE EXTERNAL CLUSTERS!
            pass
        else:
            # UPDATE INTERNAL CLUSTER (mainly for testing)
            self.data_cluster.update(samples)
        # invalid buffered decision function
        self.decision_fn_buffer = {}

    def get_hash(self, arr):
        arr.flags.writeable = False
        h = hash(arr.data)
        arr.flags.writeable = True
        return h

    def decision_function(self, samples):
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
            for i, h in enumerate(hashed):

                if h in intersec_hashes:
                    similarity_scores.append(self.decision_fn_buffer[h])
                else:
                    score = self.data_cluster.sample_set_similarity_scores(np.array([samples[i]]), self.metric)
                    similarity_scores.append(score)
                    # add to buffer
                    self.decision_fn_buffer[h] = score

        else:
            similarity_scores = self.data_cluster.sample_set_similarity_scores(samples, self.metric)
            # add to buffer
            for i, h in enumerate(hashed):
                self.decision_fn_buffer[h] = similarity_scores[i]

        similarity_scores = np.array(similarity_scores).flatten()

        return similarity_scores


class SetSimilarityHardThreshold(SetSimilarityThresholdBase):

    # hard decision threshold
    __thresh = None
    metric = None

    def __init__(self, threshold=0.3, cluster=None, metric='ABOD'):
        SetSimilarityThresholdBase.__init__(self, cluster=cluster)
        self.__thresh = threshold
        self.metric = metric

    def predict(self, samples):

        # get similarity scores
        similarity_scores = self.decision_function(samples)
        print "==== ABOD: ", ["%0.3f" % i for i in similarity_scores]
        print "==== L2: ", ["%0.3f" % i for i in self.data_cluster.class_mean_dist(samples, metric='euclidean')]

        if self.metric == 'ABOD':
            positive = similarity_scores > self.__thresh
        else:
            positive = similarity_scores < self.__thresh

        return np.array([1 if v else -1 for v in positive])



