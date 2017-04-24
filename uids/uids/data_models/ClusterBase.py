import numpy as np
from uids.utils.Logger import Logger as log
from abc import abstractmethod
from sklearn.metrics.pairwise import *
from uids.v2.set_metrics import ABOD

class ClusterBase:

    __valid_similarity_metrics = {
        'ABOD',
        'euclidean_to_mean',   # samples to set_mean < thresh
        'cosine_to_mean',      # samples to set_mean < thresh
        'euclidean_mean',      # average distance to set < thresh
        'cosine_mean'          # average distance to set < thresh
    }

    # ========= internal representation
    data = np.array([])

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

    # ------------------ set similarity metrics

    def set_similarity_score(self, samples, metric):
        """
        Calculate a single set similarity score
        :param samples:
        :param metric:
        :return:
        """
        raise ValueError('Not implemented yet!')

    def sample_set_similarity_scores(self, samples, metric):
        """
        Calculate a sample wise set similarity score
        :param samples:
        :param metric:
        :return:
        """

        if len(self.data) == 0:
            raise ValueError("Classifier has not been fitted yet. Use 'partial_fit(samples)' first!")

        # calculate sample-wise to-set similarity
        if metric == 'ABOD':
            # needs at least 3 points
            if len(self.data) < 3:
                raise ValueError("ABOD calculation needs at least 3 fitted samples!")
            return ABOD.get_score(samples, reference_set=self.data)
        elif metric == 'euclidean_to_mean':
            return self.class_mean_dist(samples, metric='euclidean')
        elif metric == 'cosine_to_mean':
            return self.class_mean_dist(samples, metric='cosine')
        elif metric == 'euclidean_mean':
            return self.mean_dist(samples, metric='euclidean')
        elif metric == 'cosine_mean':
            return self.mean_dist(samples, metric='cosine')
        else:
            raise ValueError("Invalid metric. Select from: {}".format(self.__valid_similarity_metrics))

    # ------------------ general metrics

    def mean_dist(self, samples, metric='cosine'):
        """
        :param samples: test samples
        :param metric: distance metric
        :return: Average distance between class and sample data
        """
        dist = np.mean(pairwise_distances(samples, self.data, metric=metric), axis=1)
        if metric == 'euclidean':
            dist = np.square(dist)
        return dist

    def class_mean_dist(self, samples, metric='cosine'):
        """
        :param samples: test samples
        :param metric: distance metric
        :return: Distance to class mean for every sample
        """
        class_mean = np.mean(self.data, axis=0)
        dist = pairwise_distances(class_mean.reshape(1, -1), samples, metric=metric)[0]
        if metric == 'euclidean':
            dist = np.square(dist)
        return dist

    def mean(self):
        if len(self.data) > 0:
            return np.mean(self.data, axis=0).reshape(1, -1)
        else:
            return None
