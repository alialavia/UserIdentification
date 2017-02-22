import numpy as np
from Queue import Queue
from threading import Thread, Lock
from uids.utils.Logger import Logger as log
from abc import abstractmethod


class EnsembleClassifierBase:
    """
    Ensemble Classifier comprised of multiple One Class Classifiers.

    Functionality:
    _________________________________________
    - Classifier Generation (Initialization)
    - Classifier (re-)training in threads
    """

    STATUS = 0

    # status
    STATUS_CLEAN = {
        0: 'shutdown',
        1: 'running'
    }

    # classifier instances (Ensemble)
    classifiers = {}        # classifier instances
    classifier_states = {}  # number of trainings/updates classifiers have received
    nr_classes = 0

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
        self.nr_classes += 1
        self.classifier_states[class_id] = 0

        # add samples to update stack
        self.classifier_update_stacks[class_id] = class_samples
        # directly train classifier
        return self.train_classifier(class_id)

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

        A Classifier needs the following methods:
        - fit(samples) (partial or regular)
        - predict(samples)
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


class EnsembleClassifierTypeA(EnsembleClassifierBase):
    """
    Implements decision/class identification method based on ensemble output

    Functionality:
    _________________________________________
    - Predict specific Class or Unknown based on ensemble output
    - Compute accuracy/probability measure of ensemble classification
    """

    __decision_function = []
    __decision_nr_samples = 0
    __verbose = False

    # database connection
    p_user_db = None

    # ---- class prediction threshold

    __min_valid_samples = 0.5   # unused

    # TODO: tune these parameters according to comparison with LFW - maybe adaptive threshold
    __class_thresh = 0.6       # at least X% of samples must uniquely identify a person
    __novelty_thresh = 0.2

    def __init__(self, user_db_, classifier_type):
        EnsembleClassifierBase.__init__(self, classifier_type)
        # link database
        self.p_user_db = user_db_

    # ------- ensemble prediction

    def prediction_proba(self, user_id):

        # loop through all classifiers and get individual
        # if self.CLASSIFIER == 'ABOD' or self.CLASSIFIER == 'IABOD':
        total_proba = 1
        cls_scores, class_ids = self.__decision_function

        # new user
        if user_id == -1:
            # probability that it is none of the users
            if self.nr_classes == 0:
                return 1

            for uid, clf in self.classifiers.iteritems():
                total_proba *= 1.-clf.get_proba()
            return total_proba

        for uid, clf in self.classifiers.iteritems():
            if uid == user_id:
                total_proba *= clf.get_proba()
            else:
                total_proba *= (1.-clf.get_proba())

        # loop through other classifiers
        return total_proba

    # def prediction_proba(self, user_id):
    #
    #     # loop through all classifiers and get individual
    #     # if self.CLASSIFIER == 'ABOD' or self.CLASSIFIER == 'IABOD':
    #     total_proba = 1
    #     cls_scores, class_ids = self.__decision_function
    #
    #     # new user
    #     if user_id == -1:
    #         # probability that it is none of the users
    #         if self.nr_classes == 0:
    #             return 1
    #
    #         for s in cls_scores:
    #             total_proba *= (self.__decision_nr_samples - s) / float(self.__decision_nr_samples)
    #         return total_proba
    #
    #     # is regular user
    #     for index, uid in enumerate(class_ids):
    #         if uid == user_id:
    #             # target classifier: probability that it is this class
    #             total_proba *= cls_scores[index] / float(self.__decision_nr_samples)
    #         else:
    #             # probability that it is not this class
    #             total_proba *= (self.__decision_nr_samples - cls_scores[index]) / float(self.__decision_nr_samples)
    #
    #     # loop through other classifiers
    #     return total_proba

    def __predict(self, samples):
        """predict classes: for each sample on every class, tells whether or not (+1 or -1) it belongs to class"""
        predictions = []
        class_ids = []

        with self.training_lock:
            for class_id, __clf in self.classifiers.iteritems():
                class_ids.append(class_id)
                predictions.append(__clf.predict(samples))
        return np.array(predictions), np.array(class_ids)

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

        # calc nr of positive class detections
        cls_scores = (predictions > 0).sum(axis=1)
        self.__decision_function = cls_scores, class_ids
        nr_samples = len(samples)
        self.__decision_nr_samples = nr_samples

        log.info('cl', "Classifier scores: {} | max: {}".format(cls_scores, nr_samples))

        # no classes detected at all - novelty
        if len(cls_scores[cls_scores <= self.__novelty_thresh*nr_samples]) == len(cls_scores):
            return -1

        identification_mask = cls_scores >= self.__class_thresh * nr_samples
        ids = class_ids[identification_mask]
        if len(ids) > 0:

            # multiple possible detection - invalid samples
            if len(ids) > 1:

                # use average to-class-distance to select best choice
                mean_dist_cosine = []
                mean_dist_euclidean = []
                for class_id in ids:
                    mean_dist_cosine.append(self.classifiers[class_id].mean_dist(samples))
                    mean_dist_euclidean.append(self.classifiers[class_id].mean_dist(samples, 'euclidean'))


                id_index_cosine = mean_dist_cosine.index(min(mean_dist_cosine))
                id_index_euclidean = mean_dist_euclidean.index(min(mean_dist_euclidean))

                log.severe("Samples are inambiguous. Classes: {}".format(ids))
                log.severe("IDCOS: {} | meandist cosine: {}".format(int(ids[id_index_cosine]), mean_dist_cosine))
                log.severe("IDEUC: {} | meandist euclidean: {}".format(int(ids[id_index_euclidean]), mean_dist_euclidean))

                for class_id in ids:
                    print self.classifiers[class_id].class_mean_dist(samples, 'cosine')

                mean_dist_cosine = np.array(mean_dist_cosine)
                if np.sum((mean_dist_cosine - min(mean_dist_cosine)) < 0.05) > 1:
                    print "------------- samples discarged - inambiguous"
                    return None

                return int(ids[id_index_cosine])
                # return None

            # single person identified - return id
            return int(ids[0])
        else:
            # samples unclear
            return None

    # ------- parameters

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

    # ------- deprecated

    def __dep_predict_sum(self, samples):
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
            probabilities.append(len(p[p > 0]) / float(len(samples)))
            class_ids.append(class_id)
        return np.array(probabilities), np.array(class_ids)

    def prediction_proba_old(self, user_id):
        total_proba = 1
        # is new user
        if user_id == -1:
            for uid in range(1, self.nr_classes + 1):
                dec_fn = self.decision_function(uid)
                if dec_fn < 0:
                    total_proba *= abs(dec_fn / float(self.__decision_nr_samples))
                else:
                    total_proba *= 1 - dec_fn / float(self.__decision_nr_samples)
            return total_proba

        # is regular user
        for uid in range(1, self.nr_classes + 1):
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

