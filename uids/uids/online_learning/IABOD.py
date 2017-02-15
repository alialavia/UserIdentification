from uids.online_learning.ABOD import ABOD
from uids.utils.Logger import Logger as log
from uids.utils.HullCluster import HullCluster
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class IABOD(ABOD):

    # todo: refactor - avoid coping data from HullCluster to class
    # (originally needed for superclass to access it)
    data = []
    __verbose = False
    data_cluster = HullCluster()

    def __init__(self):
        ABOD.__init__(self)

    def partial_fit(self, samples):
        if len(self.data) == 0:
            # init on first call
            self.fit(samples)
        self.data_cluster.update(samples)
        self.data = self.data_cluster.get_data()

    def mean_dist(self, samples, metric='cosine'):
        return np.mean(pairwise_distances(samples, self.data, metric=metric))
