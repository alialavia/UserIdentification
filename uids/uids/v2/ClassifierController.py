import numpy as np
from uids.utils.Logger import Logger as log
from uids.data_models.MeanShiftCluster import MeanShiftCluster
from sklearn.metrics.pairwise import *
from uids.v2.set_metrics import *


class BaseMetaController:

    def __init__(self):
        pass

    # --------- HELPERS
    @staticmethod
    def calc_adjacent_dist(samples):
        dist = array([np.linalg.norm(samples[i] - samples[i + 1]) for i in range(0, len(samples) - 1)])
        dist = np.square(dist)
        return dist

    # ----------- unused
    @staticmethod
    def check_abod(samples):
        """
        Assumes that we have only few outliers
        :param samples:
        :return:
        """

        abod = ABOD.get_set_score(samples)
        nr_errors = np.count_nonzero(abod < 0.15)
        # allowed errors
        if nr_errors > 0:
            return False
        return True

    # TODO: support weights - allow more deviation for low weighted samples
    @staticmethod
    def check_inter_sample_dist(samples, metric='euclidean'):

        # calc pairwise distance
        if metric == 'cosine':
            dist = pairwise_distances(samples, samples, metric='cosine')
            thresh = 0.7
        elif metric == 'euclidean':
            # dist = pairwise_distances(samples, samples, metric='euclidean')
            # dist = np.square(dist)
            dist = BaseMetaController.calc_adjacent_dist(samples)
            thresh = 1.4
        else:
            raise ValueError

        nr_errors = np.count_nonzero(dist > thresh)
        # print "nr errors: {}, max: {}".format(nr_errors, np.max(dist))

        # allowed errors
        if nr_errors > 0:
            log.severe("Inconsistent set! Inter-sample distances: {}".format(dist))
            return False
        return True


class BaseDataQueue(BaseMetaController):

    # raw CNN embeddings
    sample_queue = {}
    sample_weight_queue = {}

    __min_sample_length = 3    # at least 3 samples to build classifier
    __save_sample_length = 5   # at least 5 samples to be safe
    __save_weight_thresh = 7

    def __init__(self, min_sample_length=2, save_sample_length=5, save_weight_thresh=6):
        BaseMetaController.__init__(self)
        self.__min_sample_length = min_sample_length
        self.__save_sample_length = save_sample_length
        self.__save_weight_thresh = save_weight_thresh

    def drop_samples(self, tracking_id):
        self.sample_queue.pop(tracking_id, None)
        self.sample_weight_queue.pop(tracking_id, None)

    def accumulate_samples(self, tracking_id, new_samples, sample_weights=np.array([])):

        # check for set inconsistency
        samples_ok = BaseMetaController.check_inter_sample_dist(new_samples, metric='euclidean')

        if not samples_ok:
            log.severe("Identification set is inconsistent - disposing...")
            # reset queue
            self.sample_queue.pop(tracking_id, None)
            self.sample_weight_queue.pop(tracking_id, None)
            return False, np.array([]), np.array([])

        # generate placeholder weights
        if sample_weights.size == 0:
            # 5 of 10
            sample_weights = np.repeat(5, len(new_samples))

        assert len(sample_weights) == len(new_samples)

        # add samples
        if tracking_id not in self.sample_queue:
            # initialize
            self.sample_queue[tracking_id] = new_samples
            self.sample_weight_queue[tracking_id] = sample_weights
        else:
            # append
            self.sample_queue[tracking_id] = np.concatenate((self.sample_queue[tracking_id], new_samples))\
                                         if self.sample_queue[tracking_id].size \
                                         else new_samples
            self.sample_weight_queue[tracking_id] = np.concatenate((self.sample_weight_queue[tracking_id], sample_weights))\
                                         if self.sample_weight_queue[tracking_id].size \
                                         else sample_weights

        is_save_set = False

        # if set has save sample or is long enough
        if len(self.sample_queue[tracking_id]) >= self.__min_sample_length:
            if len(self.sample_queue[tracking_id]) >= self.__save_sample_length\
                    or np.count_nonzero(self.sample_weight_queue[tracking_id] >= self.__save_weight_thresh):

                # check set consistency
                samples_ok = BaseMetaController.check_inter_sample_dist(self.sample_queue[tracking_id], metric='euclidean')

                if samples_ok:
                    # set is save - allow identification
                    is_save_set = True
                else:
                    # dispose all samples
                    self.sample_queue.pop(tracking_id, None)
                    self.sample_weight_queue.pop(tracking_id, None)
                    log.severe("Set is inconsistent - disposing...")

        # TODO: return whole set or only last?
        current_samples = self.sample_queue.get(tracking_id, np.array([]))
        current_weights = self.sample_weight_queue.get(tracking_id, np.array([]))

        # not enough save samples - return what we have so far
        return is_save_set, current_samples, current_weights


class IdentificationController(BaseDataQueue):

    def __init__(self):
        BaseDataQueue.__init__(self, min_sample_length=3, save_sample_length=5, save_weight_thresh=6)


class UpdateController(BaseMetaController):

    # raw CNN embeddings
    sample_queue = {}
    sample_weight_queue = {}

    # link to classifier
    __p_multicl = None

    __queue_max_length = 10
    __inclusion_range = 5           # < __queue_max_length

    def __init__(self, p_multicl):
        BaseMetaController.__init__(self)
        self.__p_multicl = p_multicl

    def drop_samples(self, tracking_id):
        self.sample_queue.pop(tracking_id, None)
        self.sample_weight_queue.pop(tracking_id, None)

    def accumulate_samples(self, user_id, new_samples, sample_weights=np.array([])):
        """

        :param user_id:
        :param new_samples:
        :param sample_weights:
        :return:
        array : save samples (save to integrate in any way)
        bool : reset user
        int : prediction of last section
        float : confidence of last section prediction
        """

        # check for set inconsistency
        samples_ok = BaseMetaController.check_inter_sample_dist(new_samples, metric='euclidean')

        if not samples_ok:
            # no return (queue is not filled up and thus we dont have a save section)
            log.severe("Update set is inconsistent - disposing...")
            # reset queue
            self.sample_queue.pop(user_id, None)
            self.sample_weight_queue.pop(user_id, None)
            return np.array([]), True, -1, 1.

        # generate placeholder weights
        if sample_weights.size == 0:
            # 5 of 10
            sample_weights = np.repeat(5, len(new_samples))

        assert len(sample_weights) == len(new_samples)

        # add samples
        if user_id not in self.sample_queue:
            # initialize
            self.sample_queue[user_id] = new_samples
            self.sample_weight_queue[user_id] = sample_weights
        else:
            # append
            self.sample_queue[user_id] = np.concatenate((self.sample_queue[user_id], new_samples))\
                                         if self.sample_queue[user_id].size \
                                         else new_samples
            self.sample_weight_queue[user_id] = np.concatenate((self.sample_weight_queue[user_id], sample_weights))\
                                         if self.sample_weight_queue[user_id].size \
                                         else sample_weights

        target_class = -1
        confidence = 1.
        forward = np.array([])
        reset_user = False

        # do meta recognition
        # check set for inconsistencies - return only save section
        while len(self.sample_queue[user_id]) >= self.__queue_max_length:

            sample_batch = self.sample_queue[user_id][0:self.__queue_max_length]
            weight_batch = self.sample_weight_queue[user_id][0:self.__queue_max_length]

            # check set consistency
            samples_ok = BaseMetaController.check_inter_sample_dist(sample_batch, metric='euclidean')

            # predict class
            is_consistent, target_class, confidence = self.__p_multicl.predict_class(sample_batch, weight_batch)

            if samples_ok and is_consistent:
                forward = np.concatenate((forward, self.sample_queue[user_id][0:self.__inclusion_range])) \
                    if forward.size \
                    else self.sample_queue[user_id][0:self.__inclusion_range]
                # remove first x samples
                self.sample_queue[user_id] = self.sample_queue[user_id][self.__inclusion_range:]
                self.sample_weight_queue[user_id] = self.sample_weight_queue[user_id][self.__inclusion_range:]
            else:
                # dispose all samples! Whole queue!
                self.sample_queue.pop(user_id, None)
                self.sample_weight_queue.pop(user_id, None)
                log.severe("Set is inconsistent - disposing...")
                reset_user = True
                break

        # predict user if not enough samples
        if not forward.size and reset_user is False:
            is_consistent, target_class, confidence = self.__p_multicl.predict_class(self.sample_queue[user_id], self.sample_weight_queue[user_id])
            print "Not enough to forward but predict...", is_consistent, target_class, confidence

        return forward, reset_user, target_class, confidence
