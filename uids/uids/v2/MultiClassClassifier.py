import numpy as np
from uids.utils.Logger import Logger as log
from MultiClassClassifierBase import MultiClassClassifierBase
from set_metrics import ABOD

from uids.v2.HardThreshold import SetSimilarityHardThreshold
from uids.data_models.StandardCluster import StandardCluster
from uids.v2.MultiClassClassifierBase import MultiClassClassifierBase
from uids.v2.DataController import DataController
from uids.v2.ClassifierController import IdentificationController, UpdateController


class MultiCl(MultiClassClassifierBase):

    # from parent:
    # classifiers = {}

    # -------- subtask controllers

    # data storage
    data_controller = None
    # update controller: meta recognition and data inclusion gateway
    update_controller = None
    # identification controller
    id_controller = None

    def __init__(self, user_db_, classifier='SetSimilarityHardThreshold'):
        MultiClassClassifierBase.__init__(self, classifier_type=classifier)

        self.data_controller = DataController()
        self.cls_controller = UpdateController(classifier_dict=self.classifiers)
        self.id_controller = IdentificationController(classifier_dict=self.classifiers)

    # -------- standard methods

    # TODO: untested
    def validate_class(self, samples, sample_weight, target_class_id):
        """
        first use "is_guaranteed_new_class"!
        :param samples:
        :param sample_weight:
        :return: inconsistent, confidence
        """

        # individual classifier predictions (binary)
        predictions = {}
        for class_id, cls in self.classifiers.iteritems():
            predictions[class_id] = cls.predict(samples)

        is_consistent = self.check_multicl_predictions(predictions, target_class_id)

        if not is_consistent:
            return False, 1.0

        # calculate confidence for target class
        conf = self.calc_normalized_confidence(predictions[target_class_id], sample_weight)
        return True, conf

    def predict_class(self, samples, sample_weight):
        """
        first use "is_guaranteed_new_class"!
        :param samples:
        :param sample_weight:
        :return: inconsistent, class, confidence
        """

        is_consistent = True

        # individual classifier predictions (binary)
        predictions = {}
        true_pos_rate = 0.8
        true_pos_thresh = int(true_pos_rate*len(samples))
        false_pos_rate = 0.2
        false_pos_thresh = int(false_pos_rate * len(samples))


        target_classes = []

        true_positives = []
        true_positives_rates = []
        false_positives = []

        # select classes in range
        classes_in_range = self.data_controller.classes_in_range(samples=samples, metric='cosine', thresh=0.7)

        for class_id, cls in self.classifiers.iteritems():
            # only consider near classes
            if class_id in classes_in_range:
                # binary prediction
                predictions[class_id] = cls.predict(samples)

                true_positive_samples = np.count_nonzero(predictions[class_id] == 1)
                false_positive_samples = np.count_nonzero(predictions[class_id] == -1)

                # count certain detections
                if true_positive_samples > true_pos_thresh:
                    true_positives.append(class_id)
                    true_positives_rates.append(true_positive_samples)

                # count uncertain detections
                if false_positive_samples > false_pos_thresh:
                    false_positives.append(class_id)

        is_consistent = True
        target_class = None
        safe_weight = 7

        # check for inconsistent predictions
        if len(false_positives) == 0:

            if len(true_positives) == 0:
                # new class!
                target_class = -1
            elif len(true_positives) == 1:
                # target class
                target_class = true_positives[0]
            else:
                is_consistent = False # not a valid result
                best_index = np.argmax(true_positives_rates)
                target_class = true_positives[best_index]
        else:
            is_consistent = False  # not a valid result
            if len(true_positives) == 0:
                # new class!
                target_class = -1
            elif len(true_positives) == 1:
                # target class
                target_class = true_positives[0]
            else:
                best_index = np.argmax(true_positives_rates)
                target_class = true_positives[best_index]

        if is_consistent:

            # check for strong samples with inconsistent predictions
            mask = sample_weight > safe_weight
            if np.count_nonzero(mask):
                # check if safe samples predict wrong class
                for class_id, pred in predictions.iteritems():
                    if class_id == target_class:
                        # false negative
                        if np.count_nonzero(pred[mask] < 0):
                            is_consistent = False
                            break
                    else:
                        # false positive
                        if np.count_nonzero(pred[mask] > 0):
                            is_consistent = False
                            break

        # calculate confidence
        confidence = 0.

        if target_class == -1:
            # TODO: not implemented yet! Build additive score
            confidence = 1.0
        elif target_class > 0:
            confidence = self.calc_normalized_confidence(predictions[target_class], weights=sample_weight)
        else:
            confidence = 1.0

        return is_consistent, target_class, confidence

    # ----------- multicl meta recognition

    def check_multicl_predictions(self, predictions, target_class, weights=None, save_weight=7):

        if self.CLASSIFIER == 'SetSimilarityHardThreshold':
            validity, fn, fp = self.__contradictive_binary_predictions(predictions, target_class,
                                                                       weights=weights, save_weight=save_weight)
        else:
            raise Exception("Multicl Meta Recognition for continues predictions not implemented yet")

        # if important sample was false predicted
        if not validity:
            return False

        # allowed errors
        if fn > 1 or fp > 1:
            return False
        return True

    def __contradictive_binary_predictions(self, predictions, target_class, weights=None, save_weight=0):
        """used to trigger retraining"""
        fn = 0
        fp = 0
        validity = True
        for class_id, preds in predictions.iteritems():
            nr_dects = len(preds[preds > 0])
            # if wrongly predicted: class is -1 or w
            nr_samples = len(preds)
            if class_id == target_class:
                fn = nr_samples - nr_dects
            else:
                fp += nr_dects

            if target_class != class_id and weights:
                # check if there are safe samples
                mask = weights > save_weight
                if np.count_nonzero(mask):
                    # check if safe samples predict wrong class
                    if np.count_nonzero(preds[mask] > 0):
                        validity = False

        return validity, fn, fp


    def calc_abs_confidence(self, predictions, weights, max_weight):
        norm_f = 1.0/max_weight
        confidence = np.dot(predictions, np.transpose(norm_f * weights))
        return confidence

    def calc_normalized_confidence(self, predictions, weights):
        """
        :param predictions: E [0,1]
        :param weights:
        :return:
        """
        norm_f = 1.0/np.sum(weights)
        confidence = np.dot(predictions, np.transpose(norm_f * weights))
        return confidence


    # ------------------

    def get_decision_functions(self, samples):
        """
        :param samples:
        :return: new_class_guaranteed, class predictions, class scores
        """

        # no classifiers yet, predict novelty
        if not self.classifiers:
            return {}

        # predict class values
        decision_functions = {}
        for cl_id in self.classifiers.keys():
            decision_functions[cl_id] = self.classifiers[cl_id].decision_function(samples)

        # test identify always the same id
        return decision_functions

    def get_decision_functions_in_range(self, samples):
        """
        :param samples:
        :return: class scores in range
        """

        # no classifiers yet, predict novelty
        if not self.classifiers:
            return {}

        # select classes in range
        classes_in_range = self.data_controller.classes_in_range(samples=samples, metric='cosine', thresh=0.7)

        if len(classes_in_range) == 0:
            log.info('cls', "No class in range... (cosine < 0.7)")
            return {}

        # predict class values
        decision_functions = {}
        for cl_id in classes_in_range:
            decision_functions[cl_id] = self.classifiers[cl_id].decision_function(samples)

        # test identify always the same id
        return decision_functions

    def is_guaranteed_new_class(self, samples):
        if not self.classifiers:
            return True

        # select classes in range
        classes_in_range = self.data_controller.classes_in_range(samples=samples, metric='cosine', thresh=0.7)

        if len(classes_in_range) == 0:
            log.info('cls', "No class in range... (cosine < 0.7)")
            return True

        return False

    def init_new_class(self, class_id, class_samples):
        """
        Initialise a One-Class-Classifier with sample data
        :param class_id: new class id
        :param class_samples: samples belonging to the class
        :return: True/False - success
        """

        log.info('cl', "Initializing new Classifier for user ID {}".format(class_id))
        if class_id in self.classifiers:
            log.severe("Illegal reinitialization of classifier")
            return False

        # init new data model
        self.data_controller.add_samples(user_id=class_id, new_samples=class_samples)
        cluster_ref = self.data_controller.get_class_cluster(class_id)

        # init new classifier
        if self.CLASSIFIER == 'SetSimilarityHardThreshold':
            # link to data controller: similarity matching - model = data
            self.classifiers[class_id] = SetSimilarityHardThreshold(
                metric='ABOD',
                threshold=0.5,
                cluster=cluster_ref     # TODO: data model is connected - might also be separate?
            )
        elif self.CLASSIFIER == 'non-incremental':
            # link to data controller: non-incremental learner
            pass
        elif self.CLASSIFIER == 'incremental':
            # regular model. No need to pass data reference
            pass

        self.nr_classes += 1
        self.classifier_states[class_id] = 0

        # add samples to update stack
        with self.trainig_data_lock:
            self.classifier_update_stacks[class_id] = class_samples
        # directly train classifier
        return self.train_classifier(class_id)

    # -----------------------------------------------------------------

    def generate_classifier(self):
        # placeholder
        pass

    def define_classifiers(self):
        self.VALID_CLASSIFIERS = {'SetSimilarityHardThreshold'}
