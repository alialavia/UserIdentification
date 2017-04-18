import numpy as np
from uids.utils.Logger import Logger as log
from uids.data_models.StandardCluster import StandardCluster

class BaseMetaController:

    def __init__(self):
        pass

    # --------- HELPERS

    def is_consistent_set(self, samples, relative_measure=False):
        # check set for inconsistencies
        return True

    # ----------- unused

    def __contradictive_predictions(self, predictions, target_class):
        """used to trigger retraining"""
        nr_cont_samples = 0
        for class_id, col in predictions.iteritems():
            nr_dects = len(col[col > 0])
            # if wrongly predicted: class is -1 or w
            nr_samples = len(col)
            if class_id == target_class:
                nr_cont_samples += (nr_samples - nr_dects)
            else:
                nr_cont_samples += nr_dects
        return nr_cont_samples


class IdentificationController(BaseMetaController):
    # raw CNN embeddings
    sample_queue = {}
    # link to classifier dict
    p_classifiers = None

    __min_sample_length = 10
    __identification_range = 5     # < __min_sample_length

    def __init__(self, classifier_dict):
        BaseMetaController.__init__(self)
        self.p_classifiers = classifier_dict

    def drop_samples(self, tracking_id):
        self.sample_queue.pop(tracking_id, None)

    def try_to_identify(self, tracking_id, new_samples):
        # add samples
        if tracking_id not in self.sample_queue:
            # initialize
            self.sample_queue[tracking_id] = new_samples
        else:
            # append
            self.sample_queue[tracking_id] = np.concatenate((self.sample_queue[tracking_id], new_samples))\
                                         if self.sample_queue[tracking_id].size \
                                         else new_samples
        is_save_set = False
        # TODO: return whole set or only last?
        current_samples = self.sample_queue[tracking_id]

        # try to do identification
        if len(self.sample_queue[tracking_id]) >= self.__min_sample_length:
            # meta recognition
            if self.is_consistent_set(self.sample_queue[tracking_id], relative_measure=True):
                # set is save - allow identification
                is_save_set = True
            else:
                log.severe("Identification set is inconsistent - disposing...")
            # dispose all samples
            self.sample_queue.pop(tracking_id, None)

        # not enough save samples - return what we have so far
        return is_save_set, current_samples


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

