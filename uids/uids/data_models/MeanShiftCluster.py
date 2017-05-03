import numpy as np
from uids.utils.Logger import Logger as log
from uids.data_models.ClusterBase import ClusterBase
from sklearn.metrics.pairwise import *
from uids.v2.set_metrics import ABOD


class MeanShiftCluster(ClusterBase):
    """
    Notes:
        -
    """

    __max_size = None

    # ========= parameters

    # ========= internal state
    data_mean = None

    def __init__(self, max_size=40):
        ClusterBase.__init__(self)
        self.__max_size = max_size

    def mean(self):
        return self.data_mean

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
        # add data
        self.data = np.concatenate((self.data, samples)) if self.data.size else np.array(samples)
        # calculate mean
        self.data_mean = np.mean(self.data, axis=0).reshape(1, -1)
        # reduce data
        self.__reduce_after()
        # TODO: do not add points that are very near to each other


class MeanShiftPoseCluster(ClusterBase):
    __max_size = None

    # ========= parameters
    __valid_similarity_metrics = {'ABOD'}

    # ========= internal state
    data_mean = None
    poses = np.array([])
    p_weight_gen = None

    def __init__(self, p_weight_gen, max_size=40):
        ClusterBase.__init__(self)
        self.__max_size = max_size
        self.p_weight_gen = p_weight_gen

    def mean(self):
        return self.data_mean

    def __reduce_after(self, metric='cosine', reverse=True):
        # TODO: clean near sample points
        # TODO: clean samples with same pose but large variation (most likely erronomous)

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

    def update(self, samples, poses=None):
        # add data
        self.data = np.concatenate((self.data, samples)) if self.data.size else np.array(samples)
        self.poses = np.concatenate((self.poses, samples)) if self.poses.size else np.array(poses)

        # calculate mean
        self.data_mean = np.mean(self.data, axis=0).reshape(1, -1)
        # reduce data
        self.__reduce_after()

    # used in thresholding with this cluster
    def sample_set_similarity_scores(self, samples, metric):

        if len(self.data) == 0:
            raise ValueError("Classifier has not been fitted yet. Use 'partial_fit(samples)' first!")

        # calculate sample-wise to-set similarity
        if metric == 'ABOD':
            # needs at least 3 points
            if len(self.data) < 3:
                raise ValueError("ABOD calculation needs at least 3 fitted samples!")

            # select best fitting data
            abof_scores = []

            for emb in samples:
                indices, pose_separation = self.p_weight_gen.best_subset()
                abof_val = ABOD.get_score(samples, reference_set=self.data)
                abof_scores.append(abof_val)



            # get ABOD score for most relevant samples
            return abof_score, weights
        else:
            raise ValueError("Invalid metric for MeanShiftPoseCluster. Select from: {}".format(self.__valid_similarity_metrics))


