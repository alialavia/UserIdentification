import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import svm
import time
from Queue import Queue
from threading import Thread, Lock
from sklearn.preprocessing import LabelEncoder
from uids.utils.Logger import Logger as log
from abc import abstractmethod
from uids.online_learning.ABOD import ABOD


class MultiClassTree:
    """
    Goal: labeled classes, user database
    """

    """
    A Classifier needs the following methods:
    - fit(samples) (partial or regular)
    - predict(samples)

    """

    CLASSIFIER = ''
    VALID_CLASSIFIERS = {}

    classifiers = {}
    classifier_states = {}
    training_counter = {}   # number of times classifier has been trained

    __nr_classes = 0

    __verbose = False

    # status
    STATUS_CLEAN = {
        0: 'shutdown',
        1: 'running'
    }

    # multi-threaded training
    __tasks = Queue(maxsize=0)
    __num_threads = 3
    training_lock = Lock()

    # database connection
    p_user_db = None

    # ---- class prediction threshold
    # TODO: tune these parameters according to comparison with LFW - maybe adaptive threshold
    __class_thresh = 0.75       # X% of samples must be identified positively to identify person
    __confusion_thresh = 0.01   # 1% confusion chance
    __novelty_thresh = 0.01     # 1% novelty missdetection

    # ----------------------- abstract methods

    @abstractmethod
    def define_classifiers(self):
        """
        e.g.
        self.VALID_CLASSIFIERS = {'OCSVM', 'OCSVM_RBF', 'RF'}
        self.CLASSIFIER = 'OCSVM'
        """
        raise NotImplementedError("Classifier options must be defined first.")

    @abstractmethod
    def generate_classifier(self):
        """
        Generate classifier instance - e.g.:
        if self.CLASSIFIER == 'OCSVM':
            return svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
        elif self.CLASSIFIER == 'OCSVM_RBF':
            return svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        elif self.CLASSIFIER == 'RF':
            return IsolationForest(random_state=np.random.RandomState(42))
        """
        raise NotImplementedError("Classifier generation must be implemented first.")

    @abstractmethod
    def train_classifier(self, class_id):
        """
        e.g.
        samples = self.p_user_db.get_class_samples(class_id)
        self.classifiers[class_id].fit(samples)
        """
        raise NotImplementedError("Classifier training must be implemented first.")

    @abstractmethod
    def process_labeled_stream_data(self, class_id, samples):
        """
        Incorporate labeled data into the classifiers. Classifier for {class_id} must be initialized already
        """
        raise NotImplementedError("Stream processing must be implemented first.")

    def __init__(self, user_db_, classifier_type):
        # define valid classifiers
        self.define_classifiers()

        print self.VALID_CLASSIFIERS
        if classifier_type not in self.VALID_CLASSIFIERS:
            raise ValueError('Invalid Classifier "{}". You can choose between: {}'.format(classifier_type, str(list(self.VALID_CLASSIFIERS))))
        # link database
        self.p_user_db = user_db_
        # perform classifier training in tasks
        self.__deploy_classifier_trainers()

    def init_classifier(self, class_id, class_samples):
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
        self.classifiers[class_id] = self.generate_classifier()
        self.training_counter[class_id] = 0
        self.__nr_classes += 1
        self.classifier_states[class_id] = 0
        # collect the samples
        self.p_user_db.add_samples(class_id, class_samples)
        # train the classifier
        return self.train_classifier(class_id)

    # ------- ensemble prediction

    def predict(self, samples):
        """predict classes: for each sample on every class, tells whether or not (+1 or -1) it belongs to class"""
        predictions = {}

        with self.training_lock:
            for class_id, __clf in self.classifiers.iteritems():
                predictions[class_id] = __clf.predict(samples)
        return predictions

    def predict_proba(self, samples):
        """
        Predict class probabilites from samples
        :param samples: list of samples
        :return: (np.array, np.array) probabilities (positive samples/total samples per class), class ids
        """
        predictions = self.predict(samples)
        # analyze
        class_ids = []
        probabilities = []
        for class_id, p in predictions.iteritems():
            probabilities.append(len(p[p > 0])/float(len(samples)))
            class_ids.append(class_id)
        return np.array(probabilities), np.array(class_ids)

    def predict_class(self, samples):
        """
        Prediction cases:
        - Only target class is identified with ratio X (high): Class
        - Target and other class is identified with ration X (high) and Y (small): Class with small confusion
        - Multiple classes are identified with small ratios Ys: Novelty
        - No classes identified: Novelty
        :param samples:
        :return: Class ID, -1 (Novelty), None invalid dataset (multiple detections)
        """
        proba, class_ids = self.predict_proba(samples)
        mask_0 = proba > 0

        # no classes detected at all - novelty
        if len(proba[mask_0]) == 0:
            return -1

        mask_class = proba > self.__class_thresh
        nr_classes = len(proba[mask_class])

        if nr_classes > 0:
            # class detected
            if nr_classes > 1:
                # multiple classes detected
                if self.__verbose:
                    print "--- Multiple classes detected: {}".format(nr_classes)
                return None

            # count if any element, except for class is above confusion ratio
            if len(proba[(self.__confusion_thresh < proba) & (proba < self.__class_thresh)]) > 0:
                return None

            class_id_arr = class_ids[mask_class]
            return int(class_id_arr[0])

        else:
            if len(proba[proba > self.__novelty_thresh]) > 0:
                print "--- no classes detected but novelty threshold exceeded: {}".format(proba)
                return None

            return -1

    # -------- threaded classifier training

    def add_training_task(self, classifier_id):
        self.__tasks.put(classifier_id)

    def __classifier_trainer(self):
        if self.__verbose is True:
            log.info('cl', "Starting classifier training thread")
        while True:

            if self.__verbose is True:
                log.info('cl', "Begin classifier training in thread")

            # print "==== queue size: "+str(self.__tasks.qsize())
            training_id = self.__tasks.get()
            self.train_classifier(training_id)
            # reset training counter
            self.training_counter[training_id] = 0
            self.__tasks.task_done()

    def __deploy_classifier_trainers(self):
        for i in range(self.__num_threads):
            t = Thread(target=self.__classifier_trainer)
            t.daemon = True  # terminate if main thread ends
            t.start()


class OfflineMultiClassTree(MultiClassTree):

    __max_model_outliers = 1    # after x number of outlier features (per class), classifiers are retrained
    __verbose = False

    def define_classifiers(self):
        self.VALID_CLASSIFIERS = {'OCSVM', 'OCSVM_RBF', 'RF'}
        self.CLASSIFIER = 'OCSVM'

    def __init__(self, user_db_, classifier='OCSVM'):
        MultiClassTree.__init__(self, user_db_, classifier)

    def generate_classifier(self):
        if self.CLASSIFIER == 'OCSVM':
            return svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
        elif self.CLASSIFIER == 'OCSVM_RBF':
            return svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        elif self.CLASSIFIER == 'RF':
            return IsolationForest(random_state=np.random.RandomState(42))

    def train_classifier(self, class_id):
        """Retrain One-Class Classifier"""

        log.info('cl', "(Re-)training Classifier for user ID {}".format(class_id))

        if class_id not in self.classifiers:
            log.severe("Cannot train class {} without initialized classifier".format(class_id))
            return False

        samples = self.p_user_db.get_class_samples(class_id)

        # TODO: empty check for arrays
        # if not samples:
        #     if self.__verbose:
        #         print "--- Cannot train class {} without samples".format(class_id)
        #     return False

        start = time.time()
        with self.training_lock:
            self.classifiers[class_id].fit(samples)
            self.training_counter[class_id] = 0
            self.classifier_states[class_id] += 1
            if self.__verbose:
                log.info('cl', "fitting took {} seconds".format(time.time() - start))
        return True

    def process_labeled_stream_data(self, class_id, samples):
        """
        Incorporate labeled data into the classifiers. Classifier for {class_id} must be initialized already
        (retraining is done once the samples can't be explained by the model anymore)
        :param class_id: class id
        :param samples: class samples
        :return: -
        """

        log.info('cl', "Processing labeled stream data for user ID {}".format(class_id))
        class_id = int(class_id)

        if class_id not in self.training_counter:
            print "--- Class {} has not been initialized yet!".format(class_id)
            return

        # collect samples
        self.p_user_db.add_samples(class_id, samples)

        # check if incoming data explains the current model
        predictions = self.predict(samples)
        self.training_counter[class_id] += self.__contradictive_predictions(predictions, class_id)

        log.info('cl', "predictions: {}".format(predictions))
        log.info('cl', "contradictive samples accumulated: " + str(self.training_counter[class_id]))

        # trigger retraining
        if self.training_counter[class_id] >= self.__max_model_outliers:
            log.info('cl', "Retraining was triggered - adding training task")
            # threaded training
            self.add_training_task(class_id)

    # ------- Utilities

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


class OnlineMultiClassTree(MultiClassTree):

    __verbose = True
    __max_model_outliers = 1

    def define_classifiers(self):
        self.VALID_CLASSIFIERS = {'ABOD'}
        self.CLASSIFIER = 'ABOD'

    def __init__(self, user_db_, classifier='ABOD'):
        MultiClassTree.__init__(self, user_db_, classifier)

    def generate_classifier(self):
        if self.CLASSIFIER == 'ABOD':
            return ABOD()

    def train_classifier(self, class_id):
        """Retrain One-Class Classifier"""

        log.info('cl', "(Re-)training Classifier for user ID {}".format(class_id))

        if class_id not in self.classifiers:
            log.severe("Cannot train class {} without initialized classifier".format(class_id))
            return False

        samples = self.p_user_db.get_class_samples(class_id)

        if len(samples) > 100:
            log.warning("Sample size exceeding 100. No refitting.")
            return True

        start = time.time()
        with self.training_lock:
            self.classifiers[class_id].fit(samples)
            self.training_counter[class_id] = 0
            self.classifier_states[class_id] += 1
            if self.__verbose:
                log.info('cl', "fitting took {} seconds".format(time.time() - start))
        return True

    def process_labeled_stream_data(self, class_id, samples):
        """
        Incorporate labeled data into the classifiers. Classifier for {class_id} must be initialized already
        (retraining is done once the samples can't be explained by the model anymore)
        :param class_id: class id
        :param samples: class samples
        :return: -
        """

        log.info('cl', "Processing labeled stream data for user ID {}".format(class_id))
        class_id = int(class_id)

        if class_id not in self.training_counter:
            print "--- Class {} has not been initialized yet!".format(class_id)
            return

        # collect samples
        self.p_user_db.add_samples(class_id, samples)

        # check if incoming data explains the current model
        predictions = self.predict(samples)
        self.training_counter[class_id] += self.__contradictive_predictions(predictions, class_id)

        log.info('cl', "predictions: {}".format(predictions))
        log.info('cl', "contradicting samples accumulated: " + str(self.training_counter[class_id]))

        # trigger retraining
        if self.training_counter[class_id] >= self.__max_model_outliers:
            log.info('cl', "Retraining was triggered - adding training task")
            # threaded training
            self.add_training_task(class_id)

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