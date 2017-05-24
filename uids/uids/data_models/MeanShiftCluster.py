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
    __prevent_drift = False

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

    def __reduce_after(self, metric='euclidean', reverse=True):
        # TODO: clean samples with same pose but large variation (most likely erronomous)

        if len(self.data) < self.__max_size:
            return

        # prevent drift - keep frontal poses
        # keep at most 1/5 of the samples in the core to prevent drift
        # - filter only other samples
        if self.__prevent_drift:
            # take all non-frontal images (any rotation > 10 deg)
            m1 = abs(self.poses) > 10
            non_frontal_mask = np.any(m1, axis=1)
            frontal_mask = ~non_frontal_mask
            non_frontal_indices = np.flatnonzero(non_frontal_mask)
            frontal_indices = np.flatnonzero(frontal_mask)

            # indices to filter
            reduce_indices = non_frontal_indices
            save_indices = frontal_indices

            # take at minimum 4/5 non-frontal images
            save_size = int(self.__max_size * 4. / 5.)
            if len(non_frontal_indices) < save_size:
                # print reduce_indices
                # print save_indices
                # print "Save size: ", save_size
                # print "Set sizes: frontal: to add: {}, non-frontal: {}".format(save_size-len(non_frontal_indices), len(non_frontal_indices))
                reduce_indices = np.concatenate((reduce_indices, frontal_indices[0:(save_size-len(non_frontal_indices))]))
                save_indices = frontal_indices[save_size-len(non_frontal_indices):]
                # print "Samples to reduce: ", len(reduce_indices)
                # print "Remaining save samples: ", len(save_indices)

            # temporary data
            data_tmp_save = self.data[save_indices]
            pose_tmp_save = self.poses[save_indices]
            data_tmp = self.data[reduce_indices]
            pose_tmp = self.poses[reduce_indices]

            # do filtering
            dist = pairwise_distances(self.data_mean, data_tmp, metric=metric)
            dist = dist[0]
            if metric == 'euclidean':
                dist = np.square(dist)
            to_remove = len(self.data) - self.__max_size
            indices = np.arange(0, len(data_tmp))
            dist_sorted, indices_sorted = zip(*sorted(zip(dist, indices), reverse=reverse))
            indices_to_delete = indices_sorted[0:to_remove]
            log.info('cl', "Removing {} non-frontal points".format(len(indices_to_delete)))

            data_tmp = np.delete(data_tmp, indices_to_delete, axis=0)
            pose_tmp = np.delete(pose_tmp, indices_to_delete, axis=0)

            # combine filtered with save data
            self.data = np.concatenate((data_tmp, data_tmp_save))
            self.poses = np.concatenate((pose_tmp, pose_tmp_save))

        else:
            # delete X samples which are most distant
            dist = pairwise_distances(self.data_mean, self.data, metric=metric)
            dist = dist[0]
            if metric == 'euclidean':
                dist = np.square(dist)
            to_remove = len(self.data) - self.__max_size
            indices = np.arange(0, len(self.data))
            dist_sorted, indices_sorted = zip(*sorted(zip(dist, indices), reverse=reverse))
            indices_to_delete = indices_sorted[0:to_remove]
            log.info('cl', "Removing {} points".format(len(indices_to_delete)))

            # delete
            self.data = np.delete(self.data, indices_to_delete, axis=0)
            self.poses = np.delete(self.poses, indices_to_delete, axis=0)

    def get_frontal_samples(self):
        # get frontal faces - pitch and yaw smaller than 10 deg
        m1 = abs(self.poses) < 10
        m2 = np.any(m1, axis=1)
        return self.data[m2]

    def update(self, samples, poses=np.array([])):
        # check if we already have very similar samples
        if len(self.data) > 0:
            dist = pairwise_distances(samples, self.data, metric='euclidean')
            dist = np.square(dist)
            mask = dist < 0.02
            ignore_mask = np.any(mask, axis=1)

            nr_ignored = np.count_nonzero(ignore_mask)
            if np.count_nonzero(ignore_mask):
                log.info('db', "Ignoring {} samples (too similar)".format(nr_ignored))

            samples = samples[~ignore_mask]

            if len(samples) == 0:
                return

            if len(poses) > 0:
                poses = poses[~ignore_mask]

        # add data
        self.data = np.concatenate((self.data, samples)) if self.data.size else np.array(samples)
        self.poses = np.concatenate((self.poses, poses)) if self.poses.size else np.array(poses)

        # calculate mean
        self.data_mean = np.mean(self.data, axis=0).reshape(1, -1)
        # reduce data
        self.__reduce_after()

    # used in thresholding with this cluster
    def sample_set_similarity_scores(self, samples, samples_poses=np.array([]), metric='ABOD', nr_ref_samples=40):

        if len(self.data) == 0:
            raise ValueError("Classifier has not been fitted yet. Use 'partial_fit(samples)' first!")

        # calculate sample-wise to-set similarity
        if metric == 'ABOD':
            # needs at least 3 points
            if len(self.data) < 3:
                raise ValueError("ABOD calculation needs at least 3 fitted samples!")

            abof_scores = []
            confidence_scores = []
            for i, emb in enumerate(samples):

                # select best fitting data
                best_indices, pose_confidences = self.p_weight_gen.best_subset(
                    samples_poses[i], self.poses, nr_samples=nr_ref_samples, get_pose_confidence=True
                )

                abof_val = ABOD.get_score(emb.reshape(1, -1), reference_set=self.data[best_indices])
                abof_scores.append(abof_val[0])

                # mean pose pased confidence score
                confidence_scores.append(np.mean(pose_confidences))

            # get ABOD score for most relevant samples
            return abof_scores, confidence_scores
        else:
            raise ValueError("Invalid metric for MeanShiftPoseCluster. Select from: {}".format(self.__valid_similarity_metrics))


