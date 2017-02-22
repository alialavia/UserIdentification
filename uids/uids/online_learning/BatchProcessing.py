import os
from EnsembleClassifier import EnsembleClassifierTypeA
from uids.utils.Logger import Logger as log
import pickle
import time
# one class classifiers
from uids.online_learning.ABOD import ABOD
from uids.online_learning.IABOD import IABOD
from uids.online_learning.ISVM import ISVM


class BatchProcessingMultiClassTree(EnsembleClassifierTypeA):
    """
    - Processing Stream Data
    - Evaluate incoming labeled data
    """

    __verbose = True
    # unknown class data for One-VS-Rest fitting (ISVM)
    __unknown_class_data = None

    def define_classifiers(self):
        self.VALID_CLASSIFIERS = {'ABOD', 'IABOD', 'ISVM'}

    def __init__(self, user_db_, classifier='IABOD'):
        EnsembleClassifierTypeA.__init__(self, user_db_, classifier)
        if classifier == 'ISVM':
            # load lfw embeddings
            log.info('clf', 'Loading unknown class samples for ISVM classifier...')
            fileDir = os.path.dirname(os.path.realpath(__file__))
            modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')  # path to the model directory
            filename = "{}/{}".format(modelDir, "embeddings_lfw.pkl")
            if os.path.isfile(filename):
                # print filename
                with open(filename, 'r') as f:
                    embeddings = pickle.load(f)
                    f.close()

                self.__unknown_class_data = embeddings
            else:
                log.severe("Missing unknown class data... File {} not found in {}!".format(filename, modelDir))

    def generate_classifier(self):
        if self.CLASSIFIER == 'ABOD':
            return ABOD()
        elif self.CLASSIFIER == 'IABOD':
            return IABOD()
        elif self.CLASSIFIER == 'ISVM':
            return ISVM(self.__unknown_class_data)

    def train_classifier(self, class_id):
        """
        Retrain One-Class Classifiers (partial_fit)
        """

        log.info('cl', "(Re-)training Classifier for user ID {}".format(class_id))

        if class_id not in self.classifiers:
            log.severe("Cannot train class {} without creating the classifier first".format(class_id))
            return False

        start = time.time()

        with self.training_lock:
            # get update samples from stack

            # if samples available: do update with all available update samples
            # update_samples = self.classifier_update_stacks.get(class_id, []) or []

            if class_id in self.classifier_update_stacks:
                update_samples = self.classifier_update_stacks[class_id]
            else:
                update_samples = []

            if len(update_samples) > 0:

                training_before = self.classifier_states[class_id]

                if self.CLASSIFIER == 'ABOD':
                    """
                    OFFLINE Classifier: retrain with all available data
                        - Samples: Stored in user db, reloaded upon every fit
                    """
                    # instead of partial fit: add samples and do refitting over complete data
                    self.p_user_db.add_samples(class_id, update_samples)
                    samples = self.p_user_db.get_class_samples(class_id)

                    # stop
                    if len(samples) > 100:
                        log.warning("Sample size exceeding 100. No refitting.")
                    else:
                        # always use fit method (no partial fit available)
                        self.classifiers[class_id].fit(samples)
                        self.classifier_states[class_id] += 1

                elif self.CLASSIFIER == 'IABOD':
                    """
                    INCREMENTAL Methods: Use partial fit with stored update data
                        - Samples: Partially stored in ABOD Cluster
                    """
                    # partial update: partial_fit
                    self.classifiers[class_id].partial_fit(update_samples)
                    self.classifier_states[class_id] += 1

                elif self.CLASSIFIER == 'ISVM':
                    """
                    INCREMENTAL Methods: Use partial fit with stored update data
                        - Samples: Partially stored in Cluster
                    """
                    self.classifiers[class_id].partial_fit(update_samples)
                    self.classifier_states[class_id] += 1

                # empty update list if training was performed
                if self.classifier_states[class_id] - training_before == 1:
                    self.classifier_update_stacks[class_id] = []
            else:
                log.warning("No training/update samples available")

        if self.__verbose:
            log.info('cl', "fitting took {} seconds".format(time.time() - start))

        return True

    def process_labeled_stream_data(self, class_id, samples, check_update=False):
        """
        Incorporate labeled data into the classifiers. Classifier for {class_id} must be initialized already
        (retraining is done once the samples can't be explained by the model anymore)
        :param class_id: class id
        :param samples: class samples
        :param check_update: Evaluate update on the current model before using it (robust to sample pollution)
        :return: -
        """

        log.info('cl', "Processing labeled stream data for user ID {}".format(class_id))
        class_id = int(class_id)

        if class_id not in self.classifiers:
            log.severe("Class {} has not been initialized yet!".format(class_id))
            return False, 1    # force reidentification

        confidence = 1

        if check_update:
            prediction = self.predict(samples)
            # samples are not certain enough
            if prediction == None:
                return None, 1
            # calculate confidence
            confidence = self.prediction_proba(class_id)
            # detected different class
            if prediction != class_id:
                log.severe("Updating invalid class! Tracker must have switched!")
                return False, confidence    # force reidentification

        with self.training_lock:
            # add update data to stack
            if class_id not in self.classifier_update_stacks or len(self.classifier_update_stacks[class_id]) == 0:
                # create new list
                self.classifier_update_stacks[class_id] = samples
            else:
                # append
                self.classifier_update_stacks[class_id] = np.concatenate((self.classifier_update_stacks[class_id], samples))

            # request classifier update
            # Todo: only request update if available update data exceeds threshold
            self.add_training_task(class_id)

        return True, confidence

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