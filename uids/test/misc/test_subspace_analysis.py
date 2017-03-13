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
from uids.online_learning.ABOD import ABOD
from sklearn.svm import SVC

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


# ================================= #
#              Test methods


def eval_on_subspace():
    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_matthias_big.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb4 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    ref = emb1[0:40,:]
    test = emb1[40:60,:]
    ul = emb4[0:10,:]

    clf = ABOD()
    metric = 'euclidean'

    # extract 99.9% subspace
    # basis, mean = ExtractSubspace(ref, 0.9)
    basis, mean = ExtractInverseSubspace(ref, 0.7)

    print "--- reduced dimension to: {}".format(np.size(basis,1))

    # before
    sep1 = pairwise_distances(ref, test, metric=metric)
    sep2 = pairwise_distances(ref, ul, metric=metric)

    m1 = np.mean(sep1, axis=0)
    m2 = np.mean(sep2, axis=0)



    print "Original Space:"
    print "Max. dist.: inliers: {:.3f}, outliers: {:.3f}".format(sep1.max(), sep2.max())

    clf.fit(ref)
    clf.predict(test)
    clf.predict(ul)

    # ----------------------------------------------

    # project data onto subspace
    ref = ProjectOntoSubspace(ref, mean, basis)
    ul = ProjectOntoSubspace(ul, mean, basis)
    test = ProjectOntoSubspace(test, mean, basis)

    # compare
    sep1 = pairwise_distances(ref, test, metric=metric)
    sep2 = pairwise_distances(ref, ul, metric=metric)


    # meandist inliers
    print "------------------meandist to inliers-----------------------"
    print m1
    print np.mean(sep1, axis=0)
    print "Mean decrease (pos): ", m1-np.mean(sep1, axis=0)
    print "-----------------------------------------"


    # meandist outliers
    print "------------------meandist to outliers-----------------------"
    print m2
    print np.mean(sep2, axis=0)
    print "Mean decrease (neg): ", m2-np.mean(sep2, axis=0)


    clf.fit(ref)
    clf.predict(test)
    clf.predict(ul)

    print "Inlier Space:"
    print "Max. dist.: inliers: {:.3f}, outliers: {:.3f}".format(sep1.max(), sep2.max())



def eval_on_subspace2():
    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_matthias_big.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb4 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    ref = emb1[0:40,:]
    test = emb1[40:60,:]
    ul = emb4[0:10,:]

    plt.plot(np.arange(0,128), ul[0])
    # plt.plot(np.arange(0,128), ul[1])
    # plt.plot(np.arange(0,128), ul[2])


    for i in range(0,10):
        plt.plot(np.arange(0, 128), ul[i], color='blue')
    plt.plot(np.arange(0,128), test[3], color="red")
    plt.show()


# ================================= #
#              Main

if __name__ == '__main__':
    eval_on_subspace2()


