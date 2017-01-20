from sklearn import svm
import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import *
import csv
from numpy import genfromtxt
from scipy.spatial import distance as dist
import sys
import math
import pickle
from uids.utils.DataAnalysis import *

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def load_labels(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        my_data = genfromtxt(filename, delimiter=',')
        return my_data
    return None


class ABOD:
    data = None
    verbose = False
    cluster_distances = False
    # impl. : see https://github.com/MarinYoung4596/OutlierDetection
    basis = None
    mean = None

    def __init__(self):
        self.data = []

    def __calc_iters(self, knn):
        p = 0.99  # that at least one of the sets of random samples does not include an outlier
        u = 0.99  # probability that any selected data point is an inlier
        N = np.log10(1-p)/np.log10(1-np.power(u, knn))
        return N

    def train(self, data, dim_reduction=True):
        start = time.time()

        if dim_reduction is True:
            # ExtractSubspace
            self.basis, self.mean = ExtractSubspace(data, 0.999)
            print "--- reduced dimension to: {}".format(np.size(self.basis, 1))

            # project data onto subspace
            self.data = ProjectOntoSubspace(data, self.mean, self.basis)
        else:
            self.basis = None
            self.mean = None
            self.data = data

        # calculate intra-cluster distances
        self.cluster_distances = pairwise_distances(self.data, self.data, metric='euclidean')
        print "--- Classifier initialized in {}s".format("%.4f"%(time.time()-start))

    def find_threshold(self):
        pass

    def predict(self, samples):
        start = time.time()
        # dist_table = pairwise_distances(samples, samples, metric='euclidean')
        abof = self.__abof_multi(samples)
        print "--- ABOF: {} | calc time: {}s".format(["%.5f"%item for item in abof], "%.4f"%(time.time()-start))
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
            print "--- {} iterations necessary".format(nr_iters)
            vars = []
            for i in range(0,nr_iters):
                vars.append(self.__abof_multi(samples, knn=knn))
            # TODO: max, mean?
            abof = np.mean(vars, axis=0)
        print "--- ABOF: {} | calc time: {}s".format(["%.5f"%item for item in abof], "%.4f"%(time.time()-start))
        return abof

    def predict_single(self, sample):
        start = time.time()
        # dist_table = pairwise_distances(samples, samples, metric='euclidean')
        abof = self.__abof_single(self.data, sample)
        print "--- ABOF: {} | calc time: {}s".format(abof, "%.4f"%(time.time()-start))
        return abof

    # ------------PREDICTION METRICS--------------- #

    def __abof_multi(self, samples, knn=None, cosine_weighting=True):
        """
        calculate the ABOF of A = (x1, x2, ..., xn)
        pt_list = self.data (cluster)
        """

        i = 0
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
            print 'A\n', A
            print 'B\n', B
            print 'C\n', C
            print 'AB = %f, AC = %f' % (AB, AC)
            print 'AB * AC = ', dotProduct
            print '|AB| * |AC| = ', AB * AC
            sys.exit('ERROR\tangleBAC\tmath domain ERROR, |cos<AB, AC>| <= 1')
        angle = float(math.acos(cos_AB_AC_))	# <AB, AC> = arccos(cos<AB, AC>)
        return angle

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
        if self.verbose:
            print "--- Max cosine distance: {}".format(np.max(cos_dist))
        result = np.var(cos_dist, axis=1)
        if self.verbose:
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


# ================================= #
#        Test Functions

def test_ABOD():

    clf = ABOD()

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_matthias_big.pkl")
    emb4 = load_embeddings("embeddings_laia.pkl")
    emb5 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    clf.train(emb2)

    # class_sample = emb3[100,:]
    # outlier_sample = emb1[30,:]

    abod_class = clf.predict_approx(emb3)

    abod_outliers = clf.predict_approx(emb5)
    step = 0.0001
    start = 0.005
    stop = 0.6
    il = []
    ul = []
    x = np.arange(start, stop, step)
    for thresh in x:
        il.append(float(len(abod_class[abod_class<thresh]))/len(abod_class)*100.0)
        ul.append(float(len(abod_outliers[abod_outliers>thresh]))/len(abod_outliers)*100.0)

    plt.plot(x,il,color='green', label="Inliers")
    plt.plot(x,ul,color='red', label="Outliers")
    plt.title("Classification Error")
    plt.xlabel("Threshold")
    plt.ylabel("Error [%]")
    plt.legend()
    plt.show()

    #
    thresh = 0.2
    print "error il: {}/{} : {}%".format(len(abod_class[abod_class<0.2]), len(abod_class),float(len(abod_class[abod_class<0.2]))/len(abod_class)*100.0)
    print "error ul: {}/{} : {}%".format(len(abod_outliers[abod_outliers>0.2]), len(abod_outliers),float(len(abod_outliers[abod_outliers>0.2]))/len(abod_outliers)*100.0)


# ================================= #
#              Main

if __name__ == '__main__':
    test_ABOD()
