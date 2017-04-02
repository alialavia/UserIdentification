from uids.utils.DataAnalysis import *
from uids.data_models.StandardCluster import StandardCluster
from uids.data_models.ClusterBase import ClusterBase


class BinaryThreshold:

    __verbose = False

    clf = None
    thresh = 0.99

    random_data = None
    data_cluster = None

    avg = None

    def __init__(self, cluster=None):
        if cluster is None:
            self.data_cluster = StandardCluster()
        else:
            assert issubclass(cluster, ClusterBase)
            self.data_cluster = cluster

    def partial_fit(self, samples):
        self.data_cluster.update(samples)

    def class_mean_dist(self, samples, metric='cosine'):
        return self.data_cluster.class_mean_dist(samples, metric)

    def predict(self, samples, class_mean=False):

        print "--- classifying {} samples...".format(len(samples))

        # dist
        if class_mean is True:
            # Distance to class mean for every sample
            dist = self.data_cluster.class_mean_dist(samples, 'euclidean')
        else:
            dist = pairwise_distances(samples, self.data_cluster.data, metric='euclidean')

        # square
        dist_squared = np.square(dist)

        # average
        avg = np.average(dist_squared, axis=0)
        self.avg = avg

        print avg

        # threshold
        return avg < self.thresh

    def decision_function(self, samples):
        pass
