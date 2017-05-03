from uids.online_learning.ABOD import ABOD
from uids.utils.Logger import Logger as log
from uids.data_models.HullCluster import HullCluster
from uids.data_models.MeanShiftCluster import MeanShiftCluster
from uids.data_models.ClusterBase import ClusterBase
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class IABOD(ABOD):

    # todo: refactor - avoid coping data from HullCluster to class
    # (originally needed for superclass to access it)
    data = []
    __verbose = False
    __test_offline = False
    data_cluster = None

    # todo: remove test online
    def __init__(self, test_offline=False, cluster=None):
        ABOD.__init__(self)
        self.__test_offline = test_offline
        if cluster is None:
            self.data_cluster = MeanShiftCluster()
        else:
            assert issubclass(cluster, ClusterBase)
            self.data_cluster = cluster

    def fit(self, data, dim_reduction=False):
        raise NotImplementedError("Use 'partial_fit' instead of 'fit'")

    def partial_fit(self, samples):

        if self.__test_offline is True:
            if len(self.data) == 0:
                self.data = samples
            elif len(self.data) < 40:
                self.data = np.concatenate((self.data, samples))
        else:
            self.data_cluster.update(samples)
            self.data = self.data_cluster.get_data()

    def mean_dist(self, samples, metric='cosine'):
        return self.data_cluster.mean_dist(samples, metric)

    def class_mean_dist(self, samples, metric='cosine'):
        return self.data_cluster.class_mean_dist(samples, metric)

