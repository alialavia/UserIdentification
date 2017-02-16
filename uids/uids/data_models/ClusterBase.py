import numpy as np
from uids.utils.DataAnalysis import *
from uids.utils.Logger import Logger as log
from abc import abstractmethod


class ClusterBase:

    # ========= internal representation
    data = []

    def __init__(self):
        pass

    def get_data(self):
        return self.data

    @abstractmethod
    def update(self, samples):
        """
        Specifies how to update self.data with incomming samples
        """
        raise NotImplementedError("Implement Cluster Update.")

    def cluster_type(self):
        """
        Return the name of the cluster type
        e.g. return "HullCluster"
        """
        return self.__class__.__name__

    # ------------------ general metrics

    def mean_dist(self, samples, metric='cosine'):
        """
        :param samples: test samples
        :param metric: distance metric
        :return: Average distance between class and sample data
        """
        return np.mean(pairwise_distances(samples, self.data, metric=metric))

    def class_mean_dist(self, samples, metric='cosine'):
        """
        :param samples: test samples
        :param metric: distance metric
        :return: Distance to class mean for every sample
        """
        class_mean = np.mean(self.data, axis=0)
        return pairwise_distances(class_mean.reshape(1, -1), samples, metric=metric)

    def mean(self):
        if len(self.data) > 0:
            return np.mean(self.data, axis=0)
        else:
            return None
