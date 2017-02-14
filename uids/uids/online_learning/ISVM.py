from uids.utils.DataAnalysis import *
from sklearn.svm import SVC
from uids.utils.HullCluster import HullCluster


class ISVM:

    __verbose = False
    __online_mode = True

    clf = None
    uncertainty_thresh = 0.7

    random_data = None
    data_cluster = HullCluster()
    __data_full = []

    # prediction
    prediction = None
    probability = None

    def __init__(self, random_data, online_mode=False):
        # load random data
        self.random_data = random_data
        self.clf = SVC(kernel='linear', probability=True, C=1)
        self.__online_mode = online_mode

    def decision_function(self, samples):
        pass

    def get_proba(self):
        # probability that it is the class (uncertain samples not counted)
        prob = 0
        prob += np.sum(self.probability[:,1][self.prediction == 1])
        prob += np.sum(1-self.probability[:,1][self.prediction == -1])
        prob /= len(self.probability[:,1][self.prediction != 0])
        return prob

    def predict(self, samples):
        proba = self.clf.predict_proba(samples)
        self.probability = proba
        mask_1 = np.sum(proba < self.uncertainty_thresh, axis=1) == 2
        pred = np.array([-1 if r[0] > 0.5 else 1 for r in proba])
        pred[mask_1] = 0
        self.prediction = pred
        return pred

    def __fit_vs_random(self, class_data):
        label_class = np.repeat(1, np.shape(class_data)[0])
        label_unknown = np.repeat(-1, np.shape(self.random_data)[0])
        training_embeddings = np.concatenate((class_data, self.random_data))
        training_labels = np.concatenate((label_class, label_unknown))
        self.clf.fit(training_embeddings, training_labels)

    def partial_fit(self, samples):
        if self.__online_mode is True:
            self.data_cluster.update(samples)
            reduced_data = self.data_cluster.get_data()
            # refit SVM one vs random
            self.__fit_vs_random(reduced_data)
        else:
            # collect all data
            if len(self.__data_full) == 0:
                self.__data_full = samples
            else:
                self.__data_full = np.concatenate((self.__data_full, samples))
            self.__fit_vs_random(self.__data_full)


