import numpy as np
from uids.utils.Logger import Logger as log

from uids.v2.MultiClassClassifierBase import MultiClassClassifierBase

# weighted v2
from uids.v2_weighted.HardThreshold import SetSimilarityHardThreshold
from uids.v2_weighted.DataController import DataController
from uids.v2_weighted.ClassifierController import IdentificationController, UpdateController


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
        self.update_controller = UpdateController(p_multicl=self)
        self.id_controller = IdentificationController()

    # -------- standard methods

    def predict_class(self, samples, sample_poses=None):
        """
        first use "is_guaranteed_new_class"!
        :param samples:
        :param sample_poses:
        :return: inconsistent, class, confidence
        """

        # disable sample weights
        sample_weight = None

        # individual classifier predictions (binary)
        predictions = {}
        true_pos_rate = 0.7
        true_pos_thresh = int(true_pos_rate*len(samples))
        false_pos_rate = 0.4
        false_pos_thresh = int(false_pos_rate * len(samples))

        target_positive_classes = []
        true_positives_rates = []
        false_positives = []
        pos_matching_confidence = []
        neg_matching_confidence = []

        # select classes in range
        classes_in_range = self.data_controller.classes_in_range(samples=samples, metric='euclidean', thresh=1.1)

        if not classes_in_range:
            log.info('cl', "No classes in range...")
            return True, -1, 1

        for class_id, cls in self.classifiers.iteritems():
            # only consider near classes
            if class_id in classes_in_range:
                # binary prediction
                predictions[class_id], matching_confidences = cls.predict(samples, samples_poses=sample_poses)
                true_positive_samples = np.count_nonzero(predictions[class_id] == 1)

                # count certain detections
                if true_positive_samples >= true_pos_thresh:
                    target_positive_classes.append(class_id)
                    true_positives_rates.append(true_positive_samples)
                    pos_matching_confidence.append(matching_confidences)
                elif true_positive_samples >= false_pos_thresh:
                    # not true class - check if too many false positives
                    false_positives.append(class_id)
                    neg_matching_confidence.append(matching_confidences)

                # else:
                #     false_positive_samples = np.count_nonzero(predictions[class_id] == -1)
                #     # count uncertain detections
                #     if false_positive_samples >= false_pos_thresh:
                #         false_positives.append(class_id)

        is_consistent = True
        target_class = None
        safe_weight = 7

        print "... T_fp: {}, T_tp: {} |  fp: {}, tp: {}".format(false_pos_thresh, true_pos_thresh, false_positives, target_positive_classes)


        confidence = 1.0
        decision_weights = np.repeat(99., len(samples))

        # check for inconsistent predictions
        if len(false_positives) == 0:
            if len(target_positive_classes) == 0:
                # new class!
                target_class = -1
                # TODO: confidence not implemented yet! Build additive score
            elif len(target_positive_classes) == 1:
                # single target class
                target_class = target_positive_classes[0]
                decision_weights = pos_matching_confidence[0]
            else:
                # multiple target classes
                is_consistent = False   # not a valid result
                best_index = np.argmax(true_positives_rates)
                target_class = target_positive_classes[best_index]
                decision_weights = pos_matching_confidence[best_index]
        else:
            is_consistent = False  # not a valid/safe result
            if len(target_positive_classes) == 0:
                # new class!
                target_class = -1
                # TODO: confidence not implemented yet! Build additive score
            elif len(target_positive_classes) == 1:
                # target class
                target_class = target_positive_classes[0]
                decision_weights = pos_matching_confidence[0]
            else:
                best_index = np.argmax(true_positives_rates)
                target_class = target_positive_classes[best_index]
                decision_weights = pos_matching_confidence[best_index]

        # TODO: not active right now
        # if is_consistent and False:
        #     # check for strong samples with inconsistent predictions
        #     mask = sample_weight > safe_weight
        #     if np.count_nonzero(mask):
        #         # check if safe samples predict wrong class
        #         for class_id, pred in predictions.iteritems():
        #             if class_id == target_class:
        #                 # false negative
        #                 if np.count_nonzero(pred[mask] < 0):
        #                     is_consistent = False
        #                     break
        #             else:
        #                 # false positive
        #                 if np.count_nonzero(pred[mask] > 0):
        #                     is_consistent = False
        #                     break

        # calculate confidence

        # print "decision_weights: ", decision_weights
        # print "pos_matching_confidence: ", pos_matching_confidence

        print "==== decision_weights: ", ["%0.1f" % i for i in decision_weights]

        if target_class == -1:
            # TODO: not implemented yet! Build additive score
            confidence = 1.0
        elif target_class > 0:
            # combine confidence with binary decision (weighted average)
            confidence = self.calc_normalized_positive_confidence(predictions[target_class], weights=decision_weights)
        else:
            confidence = 1.0

        print "---- Prediction: ", predictions
        print "---- Target class decision: {} / conf: {} / TP: {}, FP: {} / min. TP: {} max. FP: {}".format(target_class, confidence, len(target_positive_classes), len(false_positives), true_pos_thresh, false_pos_thresh)

        # confidence: 1...100 (full conf)
        return is_consistent, target_class, confidence

    def predict_closed_set(self, target_classes, samples):

        # choose nearest class
        mean_dist_l2, clean_ids = self.data_controller.class_mean_distances(samples, target_classes)

        if len(clean_ids) == 0:
            return None

        mean_dist_l2 = list(mean_dist_l2)
        clean_ids = list(clean_ids)
        log.info('cl', "Closed set distance scores (L2 squared): IDs {} | Class dist. : {}".format(clean_ids, mean_dist_l2))

        min_index = mean_dist_l2.index(min(mean_dist_l2))
        return clean_ids[min_index]

    # ----------- multicl meta recognition

    def check_multicl_predictions(self, predictions, target_class, weights=None, save_weight=7):
        # TODO: add pose weights

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
        assert len(predictions) == len(weights)
        predictions = np.clip(predictions, 0, 1)
        norm_f = 1.0/max_weight
        confidence = np.dot(predictions, np.transpose(norm_f * weights))
        return confidence

    def calc_normalized_positive_confidence(self, predictions, weights=None):
        """
        :param predictions: E [0,1]
        :param weights:
        :return: sum of predictions*weights
        """
        if weights is None:
            weights = np.repeat(1, len(predictions))
        assert len(predictions) == len(weights)
        predictions = np.clip(predictions, 0, 1)
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
        classes_in_range = self.data_controller.classes_in_range(samples=samples, metric='euclidean', thresh=1.1)

        if len(classes_in_range) == 0:
            log.info('cls', "No class in range... (cosine < 0.7)")
            return True

        return False

    def init_new_class(self, class_id, class_samples, sample_poses):
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
        self.data_controller.add_samples(user_id=class_id, new_samples=class_samples, new_poses=sample_poses)
        cluster_ref = self.data_controller.get_class_cluster(class_id)

        # init new classifier
        if self.CLASSIFIER == 'SetSimilarityHardThreshold':
            # link to data controller: similarity matching - model = data
            self.classifiers[class_id] = SetSimilarityHardThreshold(
                metric='ABOD',
                threshold=0.3,
                nr_compaired_samples=40,    # select 40 best samples for comparison
                cluster=cluster_ref         # linked data model
            )
        else:
            raise NotImplementedError('This classifier is not implemented yet!')

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
