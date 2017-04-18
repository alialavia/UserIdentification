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

    sample_buffer = []
    decision_fn_buffer = []
    cluster_timestamp = None
    decision_fn_timestamp = None

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

        self.cluster_timestamp = time.time()

    def decision_function(self, samples):
        """
        Distance of the samples X to the target class distribution
        :param samples:
        :return:
        """

        recalc = True

        if len(self.decision_fn_buffer) > 0 and self.decision_fn_timestamp > self.cluster_timestamp:
            if np.array_equal(samples, self.sample_buffer):
                recalc = False

        if recalc:
            similarity_scores = self.data_cluster.sample_set_similarity_scores(samples, self.metric)
            self.sample_buffer = samples
            self.decision_fn_buffer = similarity_scores
            self.decision_fn_timestamp = time.time()
        else:
            similarity_scores = self.decision_fn_buffer

        return similarity_scores


class SetSimilarityHardThreshold(SetSimilarityThresholdBase):

    # hard decision threshold
    __thresh = None
    metric = None

    def __init__(self, threshold=0.99, cluster=None, metric='ABOD'):
        SetSimilarityThresholdBase.__init__(self, cluster=cluster)
        self.__thresh = threshold
        self.metric = metric

    def predict(self, samples):
        # get similarity scores
        similarity_scores = self.decision_function(samples)
        below_thresh = similarity_scores < self.__thresh
        return [1 if v else -1 for v in below_thresh], similarity_scores



