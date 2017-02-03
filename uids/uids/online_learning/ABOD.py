from sklearn import svm
import random
import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import *
import csv
from numpy import genfromtxt
from scipy.spatial import distance as dist
import sys
import math
from uids.utils.DataAnalysis import *
from uids.utils.Logger import Logger as log


class ABOD:
    data = None
    __verbose = False
    cluster_distances = False
    # impl. : see https://github.com/MarinYoung4596/OutlierDetection
    basis = None
    mean = None
    __verbose = False

    # thresholding
    threshold = 0.2
    uncertainty_bandwidth = 0.1 # uncertain region: T-b/2 ... T+b/2

    def __init__(self):
        self.data = []

    def __calc_iters(self, knn):
        p = 0.99  # that at least one of the sets of random samples does not include an outlier
        u = 0.99  # probability that any selected data point is an inlier
        N = np.log10(1-p)/np.log10(1-np.power(u, knn))
        return N

    def fit(self, data, dim_reduction=False):
        """
        TODO: precomputed PCA subspace model
        :param data:
        :param dim_reduction:
        :return:
        """
        start = time.time()

        if dim_reduction is True:
            # ExtractSubspace
            self.basis, self.mean = ExtractSubspace(data, 0.999)
            if self.__verbose:
                log.info('cl', "reduced dimension to: {}".format(np.size(self.basis, 1)))

            # project data onto subspace
            self.data = ProjectOntoSubspace(data, self.mean, self.basis)
        else:
            self.basis = None
            self.mean = None
            self.data = data

        # calculate intra-cluster distances
        self.cluster_distances = pairwise_distances(self.data, self.data, metric='euclidean')

        log.info('cl', "New ABOD Classifier initialized in {}s".format("%.4f"%(time.time()-start)))

    def tune_threshold(self):
        pass

    def decision_function(self, samples):
        return self.__predict(samples)

    def predict(self, samples):
        """
        One Class prediction
        :param samples:
        :return: np.array of labels. 1: is-class, -1 is-not class, 0 sample is uncertain
        """

        print "--- Start prediction of samples: {}".format(len(samples))

        if len(self.data) == 0:
            log.severe("ABOD Cluster is not initialized! Please use the 'fit' method first.")

        # project onto subspace
        if self.basis is not None:
            samples = ProjectOntoSubspace(samples, self.mean, self.basis)

        variance = self.__predict(samples)

        return np.array([-1 if v < (self.threshold-self.uncertainty_bandwidth/2)
                         else 1 if v > (self.threshold+self.uncertainty_bandwidth/2)
                         else 0
                         for v in variance])

    def __predict(self, samples):
        start = time.time()
        # dist_table = pairwise_distances(samples, samples, metric='euclidean')
        abof = self.__abof_multi(samples)
        log.info('cl', "ABOF: {} | calc time: {}s".format(["%.5f"%item for item in abof], "%.4f"%(time.time()-start)))
        return abof

    def predict_approx(self, samples, knn=90, approx=False):
        start = time.time()

        # project onto subspace
        if self.basis is not None:
            samples = ProjectOntoSubspace(samples, self.mean, self.basis)

        # dist_table = pairwise_distances(samples, samples, metric='euclidean')
        if approx is True:
            abof = self.__abof_multi(samples, knn=knn)
        else:
            # multiple iterations
            nr_iters = int(self.__calc_iters(knn))
            log.info('cl', "iterations necessary".format(nr_iters))
            vars = []
            for i in range(0,nr_iters):
                vars.append(self.__abof_multi(samples, knn=knn))
            # TODO: max, mean?
            abof = np.mean(vars, axis=0)
        log.info('cl',
                 "ABOF: {} | calc time: {}s".format(["%.5f" % item for item in abof], "%.4f" % (time.time() - start)))
        return abof

    # ------------PREDICTION METRICS--------------- #

    def __abof_multi(self, samples, knn=None, cosine_weighting=False):
        """
        calculate the ABOF of A = (x1, x2, ..., xn)
        pt_list = self.data (cluster)
        """
        # Todo: fix cosine dist weighting
        pt_list = self.data

        if knn is not None and knn < len(self.data):
            pt_list = random.sample(pt_list, knn)

        dist_lookup = pairwise_distances(samples, pt_list, metric='euclidean')

        if cosine_weighting:
            cos_dist_lookup = pairwise_distances(samples, pt_list, metric='cosine')

        # print np.shape(dist_lookup[0])
        factors = []
        for i_sample, A in enumerate(samples):
            varList = []
            for i in range(len(pt_list)):
                B = pt_list[i]
                AB = dist_lookup[i_sample][i]
                j = 0
                for j in range(i + 1):
                    if j == i:  # ensure B != C
                        continue

                    C = pt_list[j]
                    AC = dist_lookup[i_sample][j]

                    if np.array_equal(B, C):
                        log.error("Points are equal: B == C!")
                        print "Bi/Cj: {}/{}".format(i, j)
                        sys.exit('ERROR\tangleBAC\tmath domain ERROR, |cos<AB, AC>| <= 1')

                    angle_BAC = self.__angleBAC(A, B, C, AB, AC)
                    # compute each element of variance list
                    try:
                        # apply weighting
                        if cosine_weighting:
                            tmp = angle_BAC / float(math.pow((2.0-cos_dist_lookup[i_sample][i]) * (2.0-cos_dist_lookup[i_sample][j]), 2))
                        else:
                            tmp = angle_BAC / float(math.pow(AB * AC, 2))
                    except ZeroDivisionError:
                        sys.exit('ERROR\tABOF\tfloat division by zero! Trying to predict training point?')
                    varList.append(tmp)
            factors.append(np.var(varList))
        return factors

    def __angleBAC(self, A, B, C, AB, AC):				# AB AC mold
        """
        calculate <AB, AC>
        """
        vector_AB = B - A						# vector_AB = (x1, x2, ..., xn)
        vector_AC = C - A
        mul = vector_AB * vector_AC				# mul = (x1y1, x2y2, ..., xnyn)
        dotProduct = mul.sum()					# dotProduct = x1y1 + x2y2 + ... + xnyn

        try:
            cos_AB_AC_ = dotProduct / (AB * AC) # cos<AB, AC>
        except ZeroDivisionError:
            sys.exit('ERROR\tangleBAC\tdistance can not be zero!')

        if math.fabs(cos_AB_AC_) > 1:
            if np.array_equal(B, C):
                log.error("Points are equal: B == C")
            elif np.array_equal(A, B):
                log.error("Points are equal: A == B")
            elif np.array_equal(A, C):
                log.error("Points are equal: A == C")

            print 'AB = %f, AC = %f' % (AB, AC)
            print 'AB * AC = ', dotProduct
            print '|AB| * |AC| = ', AB * AC
            sys.exit('ERROR\tangleBAC\tmath domain ERROR, |cos<AB, AC>| <= 1')
        angle = float(math.acos(cos_AB_AC_))	# <AB, AC> = arccos(cos<AB, AC>)
        return angle

    # ----------- UNUSED

    def predict_single(self, sample):
        start = time.time()
        # dist_table = pairwise_distances(samples, samples, metric='euclidean')
        abof = self.__abof_single(self.data, sample)
        print "--- ABOF: {} | calc time: {}s".format(abof, "%.4f"%(time.time()-start))
        return abof

    def __abof_single(self, pt_list, A):
        """
        calculate the ABOF of A = (x1, x2, ..., xn)
        pt_list = self.data (cluster)
        """
        i = 0
        dist_lookup = pairwise_distances(A.reshape(1,-1) , pt_list, metric='euclidean')
        dist_lookup = dist_lookup[0] # single point A
        # print np.shape(dist_lookup[0])
        varList = []
        for i in range(len(pt_list)):
            B = pt_list[i]
            AB = dist_lookup[i]
            j = 0
            for j in range(i + 1):
                if j == i:  # ensure B != C
                    continue

                C = pt_list[j]
                AC = dist_lookup[j]
                angle_BAC = self.__angleBAC(A, B, C, AB, AC)
                # compute each element of variance list
                try:
                    tmp = angle_BAC / float(math.pow(AB * AC, 2))
                except ZeroDivisionError:
                    sys.exit('ERROR\tABOF\tfloat division by zero! Trying to predict training point?')
                varList.append(tmp)

        variance = np.var(varList)
        return variance

    def __predict_mean_cosdist(self, samples, weighted=False):
        cos_dist = pairwise_distances(samples, self.data, metric='cosine')
        if weighted:
            print np.shape(cos_dist)
            print np.shape(self.weights)
            # apply weight (total weight=1)
            cos_dist = cos_dist * self.weights
            # sum sample influence
            result = np.sum(cos_dist, axis=1)
        else:
            result = np.mean(cos_dist, axis=1)
        return result

    def __predict_cosine_dist_var(self, samples):
        cos_dist = pairwise_distances(samples, self.data, metric='cosine')
        if self.__verbose:
            print "--- Max cosine distance: {}".format(np.max(cos_dist))
        result = np.var(cos_dist, axis=1)
        if self.__verbose:
            print "--- ABOD: {}".format(result)
        return result

    def __calc_weightsSVM(self, data):
        clf = svm.OneClassSVM(nu=0.01, gamma=0.15)
        clf.fit(data)
        print "---------decision"
        dec = clf.decision_function(data)
        print np.shape(dec)
        print np.shape(dec[:,0])
        print "---------decision"
        dec = dec[:,0]

        # 1 = sum(scale * weight_i) = scale * sum(weight_i)
        # scale = 1/sum(weight_i)

        # normalize weights
        dec = dec/np.sum(dec)

        # return dec
        return dec
