import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn import svm

import time
import pickle
import os
from Queue import Queue
from threading import Thread, Lock

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models', 'embedding_samples')	# path to the model directory


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


class DataManager:
    __data = {}

    def __init__(self):
        pass

    def get_data(self, data_id):
        if data_id not in self.__data:
            return None
        return self.__data[data_id]

    def add_data(self, data_id, samples):
        if data_id not in self.__data:
            self.__data[data_id] = samples
        else:
            self.__data[data_id] = np.concatenate((self.__data[data_id], samples))

    # TODO: implement intelligent storage:
    # -- save to hd if not used in a while (Memory-map files)
    # -- define memory limit
    # -- add data reduction method (clean out data set, time windows)


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
    __verbose = False

    STATUS_CLEAN = {
        0: 'shutdown',
        1: 'running'
    }

    # ---- class prediction threshold
    # TODO: tune these parameters according to comparison with LFW - maybe adaptive threshold
    __class_thresh = 0.8      # known person above thresh
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
                return -1

            # count if any element, except for class is above confusion ratio
            if len(proba[(self.__confusion_thresh < proba) & (proba < self.__class_thresh)]) > 0:
                return -1

            return class_ids[mask_class]

        else:
            if len(proba[proba > self.__novelty_thresh]) > 0:
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

    IsolationForest(random_state=np.random.RandomState(42))

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

# ================================= #
#              Main

if __name__ == '__main__':

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    clf = OneClassDetectorTree('OCSVM')

    np.random.shuffle(emb1)
    np.random.shuffle(emb2)
    np.random.shuffle(emb3)
    np.random.shuffle(emb_lfw)

    split_set = np.array_split(emb1, 6)
    training_1 = split_set[0:3]
    test_1 = split_set[3:6]

    split_set = np.array_split(emb2, 6)
    training_2 = split_set[0:3]
    test_2 = split_set[3:6]

    split_lfw = np.array_split(emb_lfw, 6)

    for i in range(3):
        if i==0:
            # add test class
            if not clf.init_classifier(1, training_1[i]):
                print "--- initialization failed"
            if not clf.init_classifier(2, training_2[i]):
                print "--- initialization failed"
        else:
            print "----PREDICTION: SET 1----------"
            print clf.predict_class(test_1[i])
            print "-------------------------------"

            clf.process_labeled_stream_data(1, training_1[i])
            clf.process_labeled_stream_data(2, training_2[i])

            print "----PREDICTION: SET 1----------"
            print clf.predict_class(test_1[i])
            print "-------------------------------"
            print "----PREDICTION: SET 2----------"
            print clf.predict_class(test_2[i])
            print "-------------------------------"

    while True:
        pass