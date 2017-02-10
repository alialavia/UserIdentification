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
from uids.online_learning.IABOD import IABOD


class MultiClassTreeBase:

    STATUS = 0

    # status
    STATUS_CLEAN = {
        0: 'shutdown',
        1: 'running'
    }

    # placeholder - implemented in specific classifier
    CLASSIFIER = ''
    VALID_CLASSIFIERS = {}

    # multi-threaded training
    __verbose = False
    __tasks = Queue(maxsize=0)
    __num_threads = 3
    training_lock = Lock()

    classifier_update_stacks = {}   # store update data till trainer is available

    def __init__(self, classifier_type):

        # define valid classifiers
        self.define_classifiers()

        if classifier_type not in self.VALID_CLASSIFIERS:
            raise ValueError('Invalid Classifier "{}". You can choose between: {}'.format(classifier_type, str(list(self.VALID_CLASSIFIERS))))

        self.CLASSIFIER = classifier_type

        # perform classifier training in tasks
        self.start_classifier_trainers()

        log.info('cl', "{} Classifier Tree initialized".format(self.CLASSIFIER))

    # -------- threaded classifier training

    def add_training_task(self, classifier_id):
        self.__tasks.put(classifier_id)

    def __classifier_trainer(self):
        if self.__verbose is True:
            log.info('cl', "Starting classifier training thread")

        while self.STATUS == 1:
            if self.__verbose is True:
                log.info('cl', "Begin classifier training in thread")

            # print "==== queue size: "+str(self.__tasks.qsize())
            training_id = self.__tasks.get()
            self.train_classifier(training_id)
            self.__tasks.task_done()

    def start_classifier_trainers(self):
        self.STATUS = 1
        for i in range(self.__num_threads):
            t = Thread(target=self.__classifier_trainer)
            t.daemon = True  # terminate if main thread ends
            t.start()

    def stop_classifier_trainers(self):
        self.STATUS = 0

    # ----------------------- abstract methods

    @abstractmethod
    def define_classifiers(self):
        """
        e.g.
        self.VALID_CLASSIFIERS = {'OCSVM', 'OCSVM_RBF', 'RF'}
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


class MultiClassTree(MultiClassTreeBase):
    """
    Goal: labeled classes, user database
    """

    """
    A Classifier needs the following methods:
    - fit(samples) (partial or regular)
    - predict(samples)

    """

    classifiers = {}    # classifier instances
    classifier_states = {}
    __decision_function = []
    __decision_nr_samples = 0
    __nr_classes = 0
    __verbose = False

    # database connection
    p_user_db = None

    # ---- class prediction threshold
    # TODO: tune these parameters according to comparison with LFW - maybe adaptive threshold
    __min_valid_samples = 0.5
    __class_thresh = 0.4       # at least X% of samples must uniquely identify a person
    __novelty_thresh = 0.4     # at least X% of samples should uniquely hint a novelty

    def __init__(self, user_db_, classifier_type):
        MultiClassTreeBase.__init__(self, classifier_type)
        # link database
        self.p_user_db = user_db_

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
        self.__nr_classes += 1
        self.classifier_states[class_id] = 0

        # add samples to update stack
        self.classifier_update_stacks[class_id] = class_samples
        # train the classifier
        return self.train_classifier(class_id)

    # ------- ensemble prediction

    def __predict(self, samples):
        """predict classes: for each sample on every class, tells whether or not (+1 or -1) it belongs to class"""
        predictions = []
        class_ids = []

        with self.training_lock:
            for class_id, __clf in self.classifiers.iteritems():
                class_ids.append(class_id)
                predictions.append(__clf.predict(samples))
        return np.array(predictions), np.array(class_ids)

    # DEPRECATED
    def predict_proba(self, samples):
        """
        Predict class probabilites from samples
        :param samples: list of samples
        :return: (np.array, np.array) probabilities (positive samples/total samples per class), class ids
        """
        predictions, class_ids = self.__predict(samples)

        # analyze
        probabilities = []

        for class_id, p in predictions.iteritems():
            probabilities.append(len(p[p > 0])/float(len(samples)))
            class_ids.append(class_id)
        return np.array(probabilities), np.array(class_ids)

    def prediction_proba(self, user_id):

        total_proba = 1
        # is new user
        if user_id == -1:
            for uid in range(1, self.__nr_classes + 1):
                dec_fn = self.decision_function(uid)
                if dec_fn < 0:
                    total_proba *= abs(dec_fn / float(self.__decision_nr_samples))
                else:
                    total_proba *= 1 - dec_fn / float(self.__decision_nr_samples)
            return total_proba

        # is regular user
        for uid in range(1, self.__nr_classes+1):
            if uid == user_id:
                # target classifier
                total_proba *= self.decision_function(uid) / float(self.__decision_nr_samples)
            else:
                dec_fn = self.decision_function(uid)

                if dec_fn < 0:
                    total_proba *= abs(dec_fn / float(self.__decision_nr_samples))
                else:
                    total_proba *= 1 - dec_fn / float(self.__decision_nr_samples)
                    log.severe("Duplicate detection.")
                    raise ValueError

        # loop through other classifiers
        return total_proba

    def decision_function(self, user_id=0):
        """
        Use "predict" first and then "decision_function" to extract the classifier votes
        :return: tree votes: min = -nr_samples, max = nr_samples
        """
        if user_id != 0:
            cls_scores, class_ids = self.__decision_function
            try:
                index = np.where(class_ids == user_id)
                return cls_scores[index]
            except:
                return -self.__decision_nr_samples
        else:
            return self.__decision_function

    def predict(self, samples):
        """
        Prediction cases:
        - Only target class is identified with ratio X (high): Class
        - Target and other class is identified with ration X (high) and Y (small): Class with small confusion
        - Multiple classes are identified with small ratios Ys: Novelty
        - No classes identified: Novelty
        :param samples:
        :return: Class ID, -1 (Novelty), None invalid samples (multiple detections)
        """

        # no classifiers yet, predict novelty
        if not self.classifiers:
            # 100% confidence
            self.__decision_function = np.array([len(samples)]), np.array([-1])
            return -1

        predictions, class_ids = self.__predict(samples)
        cls_scores = np.sum(predictions, axis=1)
        self.__decision_function = cls_scores, class_ids
        nr_samples = len(samples)
        self.__decision_nr_samples = nr_samples

        log.info('cl', "Classifier scores: {} | max: {}".format(cls_scores, nr_samples))

        # no classes detected at all - novelty
        # novelty_mask = cls_scores <= self.__novelty_thresh * nr_samples
        novelty_mask = cls_scores < 0

        if len(cls_scores[novelty_mask]) == len(cls_scores):
            return -1

        # identification_mask = cls_scores >= self.__class_thresh * nr_samples

        identification_mask = cls_scores >= 0
        ids = cls_scores[identification_mask]
        if len(ids) > 0:

            # multiple possible detection - invalid samples
            if len(ids) > 1:
                return None

            # single person identified - return id
            return int(class_ids[identification_mask][0])
        else:
            # samples unclear
            return None

    def __predict_ORIG(self, samples):

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
                # multiple classes detected - batch invalid
                if self.__verbose:
                    log.severe("Multiple classes detected: {}".format(nr_classes))
                return None

            confusion_mask = (self.__confusion_thresh < proba) & (proba < self.__class_thresh)
            # count if any element, except for class is above confusion ratio
            if len(proba[confusion_mask]) > 0:
                log.warning("Class confusion - force re-identification: {}% confusion, {}% identification, {} samples"
                            .format(proba[(self.__confusion_thresh < proba) & (proba < self.__class_thresh)],
                                    proba[mask_class],
                                    len(samples)))

                # calc pairwise distance. If small then force re-identification
                # for sample in proba[confusion_mask]:

                # Todo: implement properly
                # return None

            class_id_arr = class_ids[mask_class]
            return int(class_id_arr[0])

        else:
            if len(proba[proba > self.__novelty_thresh]) > 0:
                print "--- no classes detected but novelty threshold exceeded: {}".format(proba)
                return None

            return -1


class OnlineMultiClassTree(MultiClassTree):
    """
    - Processing Stream Data
    - Evaluate incoming labeled data
    """

    __verbose = True
    # __max_model_outliers = 1

    def define_classifiers(self):
        self.VALID_CLASSIFIERS = {'ABOD', 'IABOD'}

    def __init__(self, user_db_, classifier='IABOD'):
        MultiClassTree.__init__(self, user_db_, classifier)

    def generate_classifier(self):
        if self.CLASSIFIER == 'ABOD':
            return ABOD()
        elif self.CLASSIFIER == 'IABOD':
            return IABOD()

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

                    # first time: use fit
                    if self.classifier_states[class_id] == 0:
                        self.classifiers[class_id].fit(update_samples)
                    else:
                        # partial update: partial_fit
                        self.classifiers[class_id].partial_fit(update_samples)

                    self.classifier_states[class_id] += 1

                # empty update list
                self.classifier_update_stacks[class_id] = []
            else:
                log.warning("No training/update samples available")

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

        if class_id not in self.classifiers:
            log.severe("Class {} has not been initialized yet!".format(class_id))
            return False    # force reidentification

        prediction = self.predict(samples)

        if prediction != class_id:
            log.severe("Updating invalid class! Tracker must have switched!")
            return False    # force reidentification

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