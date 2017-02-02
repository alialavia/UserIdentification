from uids.online_learning.MultiClassTree import MultiClassTree


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
        predictions = self.decision_function(samples)
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
