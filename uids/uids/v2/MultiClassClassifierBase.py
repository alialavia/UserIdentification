import Queue
from threading import Thread, Lock
from uids.utils.Logger import Logger as log
from abc import abstractmethod
from time import sleep, time
import numpy as np


class MultiClassClassifierBase:
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
    __tasks = Queue.Queue(maxsize=0)
    __num_threads = 3

    # training data
    trainig_data_lock = Lock()
    classifier_update_stacks = {}   # store update data till trainer is available

    # training on timeout
    training_timeouts = {}  # add a timestamp, when sample is first added
    training_timeout = 10   # force training after 10sec

    def __init__(self, classifier_type):

        # define valid classifiers
        self.define_classifiers()

        if classifier_type not in self.VALID_CLASSIFIERS:
            raise ValueError('Invalid Classifier "{}". You can choose between: {}'.format(classifier_type, str(list(self.VALID_CLASSIFIERS))))

        self.CLASSIFIER = classifier_type

        # perform classifier training in tasks
        self.__start_classifier_trainers()

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
        with self.trainig_data_lock:
            self.classifier_update_stacks[class_id] = class_samples
        # directly train classifier
        return self.__train_classifier(class_id)

    def add_training_data(self, classifier_id, samples):
        with self.trainig_data_lock:
            if classifier_id in self.classifier_update_stacks:
                self.classifier_update_stacks[classifier_id] = np.concatenate((self.classifier_update_stacks[classifier_id], samples)) \
                    if self.classifier_update_stacks[classifier_id].size \
                    else samples
            else:
                self.classifier_update_stacks[classifier_id] = samples

    # -------- threaded classifier training

    def add_training_task(self, classifier_id):
        with self.__tasks.mutex:
            if classifier_id not in self.__tasks:
                self.__tasks.put(classifier_id)

    def __classifier_trainer(self):
        """
        Manually triggered classifier training
        :return:
        """
        if self.__verbose is True:
            log.info('cl', "Starting classifier training thread")

        while self.STATUS == 1:
            try:
                training_id = self.__tasks.get(False)
            except Queue.Empty:
                sleep(0.25)  # Time in seconds.
            else:
                if training_id not in self.classifiers:
                    log.severe("Cannot train class {} without creating the classifier first".format(training_id))
                else:
                    self.__train_classifier(training_id)
                self.__tasks.task_done()

    def __timeout_checker(self):
        while self.STATUS == 1:
            # check timeout trainings
            now = time()
            for cls_id in self.training_timeouts.keys():
                if now - self.training_timeouts[cls_id] > self.training_timeout:
                    # add timed out classifiers to training tasks
                    self.add_training_task(cls_id)
                    del self.training_timeouts[cls_id]
            sleep(1)  # sleep (sec)

    def stop_classifier_trainers(self):
        self.STATUS = 0

    def __start_classifier_trainers(self):
        self.STATUS = 1

        # classifier trainers
        for i in range(self.__num_threads):
            t = Thread(target=self.__classifier_trainer)
            t.daemon = True  # terminate if main thread ends
            t.start()

        # timeout checker
        t = Thread(target=self.__timeout_checker)
        t.daemon = True  # terminate if main thread ends
        t.start()


    #
    # @abstractmethod
    # def __train_classifier(self, class_id):
    #     """
    #     IMPLEMENTED IN FINAL MULTI-CLASS CLASSIFIER MODEL
    #
    #     e.g.
    #     samples = self.p_user_db.get_class_samples(class_id)
    #     self.classifiers[class_id].fit(samples)
    #     """
    #     raise NotImplementedError("Classifier training must be implemented first.")
    #

    # TODO: is this thread-safe? training + prediction at the same time?
    def __train_classifier(self, class_id):
        """
        Retrain One-Class Classifiers (partial_fit)
        """

        log.info('cl', "(Re-)training Classifier for user ID {}".format(class_id))

        # extract data
        with self.trainig_data_lock:
            # get update samples from stack
            if class_id in self.classifier_update_stacks:
                update_samples = self.classifier_update_stacks[class_id]
                # clear
                self.classifier_update_stacks[class_id] = []
            else:
                update_samples = []

        start = time()

        if len(update_samples) > 0:
            """
            INCREMENTAL Methods: Use partial fit with stored update data
                - Samples: Partially stored in Cluster
            """
            self.classifiers[class_id].partial_fit(update_samples)
            self.classifier_states[class_id] += 1
        else:
            log.warning("No training/update samples available")

        if self.__verbose:
            log.info('cl', "fitting took {} seconds".format(time() - start))

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
        IMPLEMENTED IN FINAL MULTI-CLASS CLASSIFIER MODEL

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
    def process_labeled_stream_data(self, class_id, samples):
        """
        IMPLEMENTED IN FINAL MULTI-CLASS CLASSIFIER MODEL

        Incorporate labeled data into the classifiers. Classifier for {class_id} must be initialized already
        """
        raise NotImplementedError("Stream processing must be implemented first.")

