import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import svm
import time
from Queue import Queue
from threading import Thread, Lock
from sklearn.preprocessing import LabelEncoder


class OneClassDetectorTree:
    """
    Possible Modes:
    - Classify only using target class classifier (does not block training)
    - Classify only on currently not used classes
    - Classify using all classes (current)
    ---------------
    Extensions:
    - Switch to more efficient classifier when dealing with lots of samples (OCSVM>Random Forest)
    """

    __CLASSIFIER = 'OCSVM'
    __VALID_CLASSIFIERS = {'OCSVM', 'OCSVM_RBF', 'RF'}

    __classifiers = {}
    __classifier_states = {}
    __retraining_counter = {}   # number of times classifier has been trained
    __training_data = {}
    __status = 1
    __nr_classes = 0
    __max_model_outliers = 1    # after x number of outlier features (per class), classifiers are retrained
    __verbose = True

    STATUS_CLEAN = {
        0: 'shutdown',
        1: 'running'
    }

    # ---- class prediction threshold
    # TODO: tune these parameters according to comparison with LFW - maybe adaptive threshold
    __class_thresh = 0.75       # X% of samples must be identified positively to identify person
    __confusion_thresh = 0.01   # 1% confusion chance
    __novelty_thresh = 0.01     # 1% novelty misdetection

    # ---- multi-threaded training
    __tasks = Queue(maxsize=0)
    __num_threads = 3
    __training_lock = Lock()

    def __init__(self, classifier='OCSVM'):
        if classifier not in self.__VALID_CLASSIFIERS:
            raise ValueError('Invalid Classifier. You can choose between: '+str(list(self.__VALID_CLASSIFIERS)))

        # perform classifier training in tasks
        self.__deploy_classifier_trainers()

    def init_classifier(self, class_id, class_samples):
        """
        Initialise a One-Class-Classifier with sample data
        :param class_id: new class id
        :param class_samples: samples belonging to the class
        :return: True/False - success
        """
        if class_id in self.__classifiers:
            return False
        self.__classifiers[class_id] = self.__generate_classifier()
        self.__retraining_counter[class_id] = 0
        self.__nr_classes += 1
        self.__classifier_states[class_id] = 0
        # collect the samples
        self.__collect_samples(class_id, class_samples)
        # train the classifier
        return self.__retrain(class_id)

    def predict_class(self, samples):
        """
        Prediction casses:
        - Only target class is identified with ratio X (high): Class
        - Target and other class is identified with ration X (high) and Y (small): Class with small confusion
        - Multiple classes are identified with small ratios Ys: Novelty
        - No classes identified: Novelty
        :param samples:
        :return: Class ID, None (Novelty), -1 invalid dataset (multiple detections)
        """
        proba, class_ids = self.predict_proba(samples)

        mask_0 = proba > 0

        # no classes detected at all - novelty
        if len(proba[mask_0]) == 0:
            return None

        mask_class = proba > self.__class_thresh
        nr_classes = len(proba[mask_class])

        if nr_classes > 0:
            # class detected
            if nr_classes > 1:
                # multiple classes detected
                if self.__verbose:
                    print "--- Multiple classes detected: {}".format(nr_classes)
                return -1

            # count if any element, except for class is above confusion ratio
            if len(proba[(self.__confusion_thresh < proba) & (proba < self.__class_thresh)]) > 0:
                return -1

            return class_ids[mask_class]

        else:
            if len(proba[proba > self.__novelty_thresh]) > 0:
                print "--- no classes detected but novelty threshold exceeded: {}".format(proba)
                return -1

            return None

    def predict_proba(self, samples):
        """
        Predict class probabilites from samples
        :param samples: list of samples
        :return: (np.array, np.array) probabilities (positive samples/total samples per class), class ids
        """
        predictions = self.__predict(samples)
        # analyze
        class_ids = []
        probabilities = []
        for class_id, p in predictions.iteritems():
            probabilities.append(len(p[p > 0])/float(len(samples)))
            class_ids.append(class_id)
        return np.array(probabilities), np.array(class_ids)

    def process_labeled_stream_data(self, class_id, samples):
        """
        Incorporate labeled data into the classifiers
        (retraining is done once the samples can't be explained by the model anymore)
        :param class_id: class id
        :param samples: class samples
        :return: -
        """
        self.__collect_samples(class_id, samples)
        # check if incoming data explains the current model
        predictions = self.__predict(samples)
        self.__retraining_counter[class_id] += self.__contradictive_predictions(predictions, class_id)

        # trigger retraining
        if self.__retraining_counter[class_id] >= self.__max_model_outliers:
            # threaded training
            self.__add_training_task(class_id)

    # ------- Utilities

    def __collect_samples(self, class_id, samples):
        if class_id not in self.__training_data:
            self.__training_data[class_id] = samples
        else:
            self.__training_data[class_id] = np.concatenate((self.__training_data[class_id], samples))

    def __predict(self, samples):
        """predict classes: for each sample on every class, tells whether or not (+1 or -1) it belongs to class"""
        predictions = {}
        with self.__training_lock:
            for class_id, __clf in self.__classifiers.iteritems():
                predictions[class_id] = __clf.predict(samples)
        return predictions

    def __contradictive_predictions(self, predictions, target_class=None):
        nr_cont_samples = 0
        if target_class is None:
            for col in np.array(predictions.values()).T:
                # only 1x1 and rest -1 or all -1
                # = nr elements =1 not greater than 1
                if len(col[col > 0]) > 1:
                    nr_cont_samples += 1
        else:
            for class_id, col in zip(predictions.keys(), np.array(predictions.values()).T):
                nr_dects = len(col[col > 0])
                if nr_dects > 1 or (nr_dects == 1 and class_id != target_class):
                    nr_cont_samples += 1
        return nr_cont_samples

    def __retrain(self, class_id):
        """Retrain One-Class Classifier"""

        if class_id not in self.__classifiers\
                or class_id not in self.__training_data:
            return False

        start = time.time()
        with self.__training_lock:
            self.__classifiers[class_id].fit(self.__training_data[class_id])
            self.__retraining_counter[class_id] = 0
            self.__classifier_states[class_id] += 1
        # print "fitting took {} seconds".format(time.time() - start)
        return True

    def __generate_classifier(self):
        """Generate classifier instance"""
        if self.__CLASSIFIER == 'OCSVM':
            return svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
        elif self.__CLASSIFIER == 'OCSVM_RBF':
            return svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        elif self.__CLASSIFIER == 'RF':
            return IsolationForest(random_state=np.random.RandomState(42))

    # -------- threaded classifier training

    def __add_training_task(self, classifier_id):
        self.__tasks.put(classifier_id)

    def __classifier_trainer(self):
        if self.__verbose is True:
            print "--- starting classifier training thread"
        while True:

            if self.__verbose is True:
                print "--- thread training classifier"

            # print "==== queue size: "+str(self.__tasks.qsize())
            training_id = self.__tasks.get()
            self.__retrain(training_id)
            self.__tasks.task_done()

    def __deploy_classifier_trainers(self):
        for i in range(self.__num_threads):
            t = Thread(target=self.__classifier_trainer)
            t.daemon = True  # terminate if main thread ends
            t.start()

    # -------- Not implemented yet

    def store_samples(self, class_id):
        pass

    def load_samples(self, class_id):
        pass


class OneClassDBDetectorTree:
    """
    Goal: labeled classes, user database
    """

    __CLASSIFIER = 'OCSVM'
    __VALID_CLASSIFIERS = {'OCSVM', 'OCSVM_RBF', 'RF'}

    __classifiers = {}
    __classifier_states = {}
    __retraining_counter = {}   # number of times classifier has been trained
    __training_data = {}
    __status = 1
    __nr_classes = 0
    __max_model_outliers = 1    # after x number of outlier features (per class), classifiers are retrained
    __verbose = True

    STATUS_CLEAN = {
        0: 'shutdown',
        1: 'running'
    }

    # ---- class prediction threshold
    # TODO: tune these parameters according to comparison with LFW - maybe adaptive threshold
    __class_thresh = 0.75       # X% of samples must be identified positively to identify person
    __confusion_thresh = 0.01   # 1% confusion chance
    __novelty_thresh = 0.01     # 1% novelty misdetection

    # ---- multi-threaded training
    __tasks = Queue(maxsize=0)
    __num_threads = 3
    __training_lock = Lock()

    # ---- database
    __p_user_db = None
    __label_encoder = None    # classifier label encoder

    def __init__(self, user_db_, classifier='OCSVM'):
        if classifier not in self.__VALID_CLASSIFIERS:
            raise ValueError('Invalid Classifier. You can choose between: '+str(list(self.__VALID_CLASSIFIERS)))
        # link database
        self.__p_user_db = user_db_
        # perform classifier training in tasks
        self.__deploy_classifier_trainers()

    def init_classifier(self, class_id, class_samples):
        """
        Initialise a One-Class-Classifier with sample data
        :param class_id: new class id
        :param class_samples: samples belonging to the class
        :return: True/False - success
        """
        if class_id in self.__classifiers:
            return False
        self.__classifiers[class_id] = self.__generate_classifier()
        self.__retraining_counter[class_id] = 0
        self.__nr_classes += 1
        self.__classifier_states[class_id] = 0
        # collect the samples
        self.__collect_samples(class_id, class_samples)
        # train the classifier
        return self.__retrain(class_id)

    def predict_class(self, samples):
        """
        Prediction casses:
        - Only target class is identified with ratio X (high): Class
        - Target and other class is identified with ration X (high) and Y (small): Class with small confusion
        - Multiple classes are identified with small ratios Ys: Novelty
        - No classes identified: Novelty
        :param samples:
        :return: Class ID, None (Novelty), -1 invalid dataset (multiple detections)
        """
        proba, class_ids = self.predict_proba(samples)

        mask_0 = proba > 0

        # no classes detected at all - novelty
        if len(proba[mask_0]) == 0:
            return None

        mask_class = proba > self.__class_thresh
        nr_classes = len(proba[mask_class])

        if nr_classes > 0:
            # class detected
            if nr_classes > 1:
                # multiple classes detected
                if self.__verbose:
                    print "--- Multiple classes detected: {}".format(nr_classes)
                return -1

            # count if any element, except for class is above confusion ratio
            if len(proba[(self.__confusion_thresh < proba) & (proba < self.__class_thresh)]) > 0:
                return -1

            return class_ids[mask_class]

        else:
            if len(proba[proba > self.__novelty_thresh]) > 0:
                print "--- no classes detected but novelty threshold exceeded: {}".format(proba)
                return -1

            return None

    def predict_proba(self, samples):
        """
        Predict class probabilites from samples
        :param samples: list of samples
        :return: (np.array, np.array) probabilities (positive samples/total samples per class), class ids
        """
        predictions = self.__predict(samples)
        # analyze
        class_ids = []
        probabilities = []
        for class_id, p in predictions.iteritems():
            probabilities.append(len(p[p > 0])/float(len(samples)))
            class_ids.append(class_id)
        return np.array(probabilities), np.array(class_ids)

    def process_labeled_stream_data(self, class_id, samples):
        """
        Incorporate labeled data into the classifiers
        (retraining is done once the samples can't be explained by the model anymore)
        :param class_id: class id
        :param samples: class samples
        :return: -
        """
        self.__collect_samples(class_id, samples)
        # collect samples

        self.__p_user_db.add_embeddings

        # check if incoming data explains the current model
        predictions = self.__predict(samples)
        self.__retraining_counter[class_id] += self.__contradictive_predictions(predictions, class_id)

        # trigger retraining
        if self.__retraining_counter[class_id] >= self.__max_model_outliers:
            # threaded training
            self.__add_training_task(class_id)

    # ------- Utilities

    def __predict(self, samples):
        """predict classes: for each sample on every class, tells whether or not (+1 or -1) it belongs to class"""
        predictions = {}
        with self.__training_lock:
            for class_id, __clf in self.__classifiers.iteritems():
                predictions[class_id] = __clf.predict(samples)
        return predictions

    def __contradictive_predictions(self, predictions, target_class=None):
        nr_cont_samples = 0
        if target_class is None:
            for col in np.array(predictions.values()).T:
                # only 1x1 and rest -1 or all -1
                # = nr elements =1 not greater than 1
                if len(col[col > 0]) > 1:
                    nr_cont_samples += 1
        else:
            for class_id, col in zip(predictions.keys(), np.array(predictions.values()).T):
                nr_dects = len(col[col > 0])
                if nr_dects > 1 or (nr_dects == 1 and class_id != target_class):
                    nr_cont_samples += 1
        return nr_cont_samples

    def __retrain(self, class_id):
        """Retrain One-Class Classifier"""

        if class_id not in self.__classifiers\
                or class_id not in self.__training_data:
            return False

        start = time.time()
        with self.__training_lock:
            self.__classifiers[class_id].fit(self.__training_data[class_id])
            self.__retraining_counter[class_id] = 0
            self.__classifier_states[class_id] += 1
        # print "fitting took {} seconds".format(time.time() - start)
        return True

    def __generate_classifier(self):
        """Generate classifier instance"""
        if self.__CLASSIFIER == 'OCSVM':
            return svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
        elif self.__CLASSIFIER == 'OCSVM_RBF':
            return svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        elif self.__CLASSIFIER == 'RF':
            return IsolationForest(random_state=np.random.RandomState(42))

    # -------- threaded classifier training

    def __add_training_task(self, classifier_id):
        self.__tasks.put(classifier_id)

    def __classifier_trainer(self):
        if self.__verbose is True:
            print "--- starting classifier training thread"
        while True:

            if self.__verbose is True:
                print "--- thread training classifier"

            # print "==== queue size: "+str(self.__tasks.qsize())
            training_id = self.__tasks.get()
            self.__retrain(training_id)
            self.__tasks.task_done()

    def __deploy_classifier_trainers(self):
        for i in range(self.__num_threads):
            t = Thread(target=self.__classifier_trainer)
            t.daemon = True  # terminate if main thread ends
            t.start()

    # -------- Not implemented yet

    def store_samples(self, class_id):
        pass

    def load_samples(self, class_id):
        pass



