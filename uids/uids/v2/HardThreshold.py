from uids.utils.DataAnalysis import *
from uids.data_models.StandardCluster import StandardCluster
from uids.data_models.ClusterBase import ClusterBase
from sklearn.metrics.pairwise import *


class SetSimilarityThresholdBase:

    __verbose = False
    __external_cluster = True   # has an external data model
    data_cluster = None

    def __init__(self, cluster=None, metric='ABOD'):

        if cluster is None:
            print "No data cluster linked. Using new StandardCluster."
            self.data_cluster = StandardCluster()
            self.__external_cluster = False
        else:
            self.data_cluster = cluster

        self.metric = metric

    def partial_fit(self, samples):
        if self.__external_cluster:
            # DONT UPDATE EXTERNAL CLUSTERS!
            pass
        else:
            # UPDATE INTERNAL CLUSTER (mainly for testing)
            self.data_cluster.update(samples)


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
        similarity_scores = self.data_cluster.sample_set_similarity_scores(samples, self.metric)
        below_thresh = similarity_scores < self.__thresh
        return [1 if v else -1 for v in below_thresh], similarity_scores

    def decision_function(self, samples=None):
        # subtract threshold
        return self.__thresh

