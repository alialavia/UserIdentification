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

    __classifiers = {}
    __classifier_states = {}
    __status = 1
    __training_data = {}
    __nr_classes = 0
    __retraining_counter = {}
    __max_model_outliers = 1
    __verbose = True

    STATUS_CLEAN = {
        0: 'shutdown',
        1: 'running'
    }

    # ----------- training

    __tasks = Queue(maxsize=0)
    __num_threads = 3
    __training_lock = Lock()

    def __init__(self):
        # perform classifier training in tasks
        self.__deploy_classifier_trainers()

    def init_classifier(self, class_id, class_samples):
        if class_id in self.__classifiers:
            return False
        self.__classifiers[class_id] = self.__generate_classifier()
        self.__retraining_counter[class_id] = 0
        self.__nr_classes += 1
        self.__classifier_states[class_id] = 0
        # collect the samples
        self.collect_samples(class_id, class_samples)
        # train the classifier

        return self.__retrain(class_id)

    def predict_class(self, samples):
        predictions = self.__predict(samples)
        # analyze
        class_ids = []
        probabilities = []
        for class_id, p in predictions.iteritems():
            probabilities.append(len(p[p > 0])/float(len(samples)))
            class_ids.append(class_id)
        return probabilities, class_ids

    def process_labeled_stream_data(self, class_id, samples):

        self.collect_samples(class_id, samples)
        # check if incoming data explains the current model
        predictions = self.__predict(samples)
        self.__retraining_counter[class_id] += self.__contradictive_predictions(predictions, class_id)

        # trigger retraining
        if self.__retraining_counter[class_id] >= self.__max_model_outliers:
            self.__retrain(class_id)

    def collect_samples(self, class_id, samples):
        # store data
        if class_id not in self.__training_data:
            self.__training_data[class_id] = samples
        else:
            self.__training_data[class_id] = np.concatenate((self.__training_data[class_id], samples))

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
        print "fitting took {} seconds".format(time.time() - start)
        return True

    def __generate_classifier(self):
        """Generate classifier instance"""
        return svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)

    # -------- threaded classifier training

    def __add_training_task(self, classifier_id):
        self.__tasks.put(classifier_id)
        self.__tasks.task_done()

    def __classifier_trainer(self):
        if self.__verbose is True:
            print "--- starting classifier training thread"
        while True:
            training_id = self.__tasks.get(True)
            self.__retrain(training_id)
            self.__tasks.task_done()

    def __deploy_classifier_trainers(self):
        for i in range(self.__num_threads):
            t = Thread(target=self.__classifier_trainer)
            t.daemon = True # terminate if main thread ends
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

    clf = OneClassDetectorTree()

    np.random.shuffle(emb1)
    np.random.shuffle(emb2)
    np.random.shuffle(emb3)

    split_set = np.array_split(emb1, 6)
    training_1 = split_set[0:3]
    test_1 = split_set[3:6]

    split_set = np.array_split(emb2, 6)
    training_2 = split_set[0:3]
    test_2 = split_set[3:6]

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
            print "----PREDICTION: SET 2----------"
            print clf.predict_class(test_2[i])
            print "-------------------------------"
