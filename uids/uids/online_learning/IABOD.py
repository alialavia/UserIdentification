from uids.online_learning.ABOD import ABOD
from uids.utils.Logger import Logger as log
from uids.data_models.HullCluster import HullCluster
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class IABOD(ABOD):

    # todo: refactor - avoid coping data from HullCluster to class
    # (originally needed for superclass to access it)
    data = []
    __verbose = False
    __test_offline = False
    data_cluster = HullCluster(knn_removal_thresh=0, inverted=False)

    def __init__(self, test_offline=True):
        ABOD.__init__(self)
        self.__test_offline = test_offline

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
        return np.mean(pairwise_distances(samples, self.data, metric=metric))

    def class_mean_dist(self, samples, metric='cosine'):
        class_mean = np.mean(self.data, axis=0)
        return pairwise_distances(class_mean.reshape(1,-1), samples, metric=metric)

