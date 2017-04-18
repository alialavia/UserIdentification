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

    def predict(self, samples):

        # no classifiers yet, predict novelty
        if not self.classifiers:
            return -1, 1.0

        # select classes in range
        classes_in_range = self.data_controller.classes_in_range(samples=samples, metric='cosine', thresh=0.7)

        if len(classes_in_range) == 0:
            return -1, 1.0

        # predict class values
        predictions = []
        for cl_id in classes_in_range:
            predictions.append(self.classifiers[cl_id].predict(samples))

        # test identify always the same id
        return 1337, 1.0

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
                threshold=0.7,
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
        return self.__train_classifier(class_id)

    def generate_classifier(self):
        # placeholder
        pass

    def define_classifiers(self):
        self.VALID_CLASSIFIERS = {'SetSimilarityHardThreshold'}
