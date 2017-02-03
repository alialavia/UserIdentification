import numpy as np
from sklearn.neighbors import KDTree
from uids.utils.Logger import Logger as log


class KNFilter:

    """
    Example:
    emb3 = load_embeddings("embeddings_matthias_big.pkl")
    # np.random.shuffle(emb3)
    f = KNFilter(emb3[0:5, :], k=3)
    print np.shape(f.filter_x_samples(2))

    """

    dist = None     # knn distances
    ind = None      # knn indices
    k = None        # number of neighbours (including self)

    __thresh = 0.3  #
    __data = None   # data link

    __verbose = False

    def __init__(self, samples, k=3, threshold=0.3, metric='euclidean'):
        kdt = KDTree(samples, leaf_size=30, metric=metric)
        self.dist, self.ind = kdt.query(samples, k=k, return_distance=True)
        # remove distance to same point
        self.dist = self.dist[:, 1:]
        self.__data = samples
        self.k = k
        self.__thresh = threshold

    def filtered(self):
        print self.dist

    def filter_by_thresh(self):
        """both connecting points in threshold"""
        pts_in_thresh = np.sum(self.dist < self.__thresh, axis=1)
        filtered = self.__data[pts_in_thresh == (self.k-1)]
        print filtered

    def filter_x_samples(self, nr_samples):

        if self.k != 3:
            log.error("METHOD NOT IMPLEMENTED FOR K != 3!")
            raise

        dist_sum = np.sum(self.dist, axis=1)

        # sort by sum value
        indices = np.arange(0, len(dist_sum))
        min_dist_sum, indices = zip(*sorted(zip(dist_sum, indices)))
        min_dist_sum = np.array(min_dist_sum)
        indices = np.array(indices)
        # sort values accordingly
        dist_sorted = np.array([self.dist[i, :] for i in indices])

        if self.__verbose:
            print self.dist
            print "---- Dist array sorted by sum:"
            print dist_sorted
            print "Indices sorted by sum: {}".format(indices)

        # sum of values below threshold
        mask1 = min_dist_sum < self.__thresh * 2
        if self.__verbose:
            print "Min dist sum: {}".format(min_dist_sum)
            print "Mask1: {}".format(mask1)

        # both values below single threshold
        diff = np.absolute(dist_sorted[:, 0] - dist_sorted[:, 1])
        mask2 = diff < self.__thresh

        comb_mask = mask1*mask2

        # filter indices in order of increasing sum
        indices_prefiltered = indices[comb_mask]

        indices_to_delete = indices_prefiltered[0:nr_samples]

        # filter first X samples
        return np.delete(self.__data, indices_to_delete, axis=0)
