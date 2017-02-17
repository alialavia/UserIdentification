import numpy as np
from uids.utils.Logger import Logger as log
from uids.data_models.ClusterBase import ClusterBase
from sklearn.metrics.pairwise import *


class StandardCluster(ClusterBase):
    """
    Notes:
        -
    """

    __max_size = None

    # ========= parameters

    # ========= internal state
    data = []
    data_mean = None

    def __init__(self, max_size=70):
        ClusterBase.__init__(self)
        self.__max_size = max_size

    def __reduce_after(self, metric='cosine', reverse=True):
        if len(self.data) < self.__max_size:
            return

        # delete X samples which are most distant
        dist = pairwise_distances(self.data_mean, self.data, metric=metric)
        dist = dist[0]
        to_remove = len(self.data) -self.__max_size
        indices = np.arange(0, len(self.data))
        dist_sorted, indices_sorted = zip(*sorted(zip(dist, indices), reverse=reverse))

        # delete X samples which are most distant
        indices_to_delete = indices_sorted[0:to_remove]

        log.info('cl', "Removing {} points".format(len(indices_to_delete)))

        # delete
        self.data = np.delete(self.data, indices_to_delete, axis=0)

    def update(self, samples):
        if len(self.data) == 0:
            self.data = samples
        else:
            self.data_mean = np.mean(self.data, axis=0).reshape(1, -1)
            # opt. 2
            self.data = np.concatenate((self.data, samples))
            # reduce data
            self.__reduce_after()
