import numpy as np
from uids.utils.Logger import Logger as log
from uids.data_models.StandardCluster import StandardCluster
from sklearn.metrics.pairwise import *
from uids.v2.set_metrics import *

class BaseMetaController:

    def __init__(self):
        pass

    # --------- HELPERS

    def is_consistent_set(self, samples):
        # check set for inconsistencies
        return True

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
            dist = pairwise_distances(samples, samples, metric='euclidean')
            dist = np.square(dist)
            thresh = 0.99
        else:
            raise ValueError

        nr_errors = np.count_nonzero(dist > thresh)
        # print "nr errors: {}, max: {}".format(nr_errors, np.max(dist))

        # allowed errors
        if nr_errors > 0:
            return False
        return True


class IdentificationController(BaseMetaController):
    # raw CNN embeddings
    sample_queue = {}
    sample_weight_queue = {}

    # link to classifier dict
    p_classifiers = None

    __min_sample_length = 2    # at least 2 samples to build classifier
    __save_sample_length = 5   # at least 5 samples to be safe

    def __init__(self, classifier_dict):
        BaseMetaController.__init__(self)
        self.p_classifiers = classifier_dict

    def drop_samples(self, tracking_id):
        self.sample_queue.pop(tracking_id, None)
        self.sample_weight_queue.pop(tracking_id, None)

    def accumulate_samples(self, tracking_id, new_samples, sample_weights=np.array([]), save_threshold=7):

        # check for set inconsistency
        samples_ok = BaseMetaController.check_inter_sample_dist(new_samples, metric='euclidean')

        if not samples_ok:
            log.severe("Identification set is inconsistent - disposing...")
            # reset queue
            self.sample_queue.pop(tracking_id, None)
            self.sample_weight_queue.pop(tracking_id, None)
            return False

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
                    or np.count_nonzero(self.sample_weight_queue[tracking_id] >= save_threshold):

                # check set consistency
                samples_ok = BaseMetaController.check_inter_sample_dist(self.sample_queue[tracking_id], metric='euclidean')

                if samples_ok:
                    # set is save - allow identification
                    is_save_set = True
                else:
                    # dispose all samples
                    self.sample_queue.pop(tracking_id, None)
                    self.sample_weight_queue.pop(tracking_id, None)
                    log.severe("Identification set is inconsistent - disposing...")

        # TODO: return whole set or only last?
        current_samples = self.sample_queue.get(tracking_id, np.array([]))
        current_weights = self.sample_weight_queue.get(tracking_id, np.array([]))

        # not enough save samples - return what we have so far
        return is_save_set, current_samples, current_weights


class UpdateController(BaseMetaController):

    # raw CNN embeddings
    sample_queue = {}
    # link to classifier dict
    p_classifiers = None

    __queue_max_length = 10
    __inclusion_range = 5           # < __queue_max_length

    def __init__(self, classifier_dict):
        BaseMetaController.__init__(self)
        self.p_classifiers = classifier_dict

    def accumulate_save_samples(self, user_id, new_samples):
        # add samples
        if user_id not in self.sample_queue:
            # initialize
            self.sample_queue[user_id] = new_samples
        else:
            # append
            self.sample_queue[user_id] = np.concatenate((self.sample_queue[user_id], new_samples))\
                                         if self.sample_queue[user_id].size \
                                         else new_samples

        current_samples = self.sample_queue[user_id]

        # do meta recognition
        # check set for inconsistencies - return only save section
        forward = np.array([])
        while self.sample_queue[user_id] >= self.__queue_max_length:
            if self.is_consistent_set(self.sample_queue[user_id][0:self.__queue_max_length]):
                forward = np.concatenate((forward, self.sample_queue[user_id][0:self.__inclusion_range])) \
                    if forward.size \
                    else self.sample_queue[user_id][0:self.__inclusion_range]
                # remove first x samples
                self.sample_queue[user_id] = self.sample_queue[user_id][self.__inclusion_range:]
            else:
                # dispose all samples! Whole queue!
                self.sample_queue[user_id] = np.array([])
                forward = np.array([])

        # save section is included - possibly compromised data is dropped
        if forward.size:
            # forward to data controller
            print "forwarding {} samples to data controller".format(len(forward))
            return True, forward
        else:
            print "no safe data to forward"
            return False, current_samples

