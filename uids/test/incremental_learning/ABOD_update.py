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
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

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


class ABODOnline(ABOD):



    def __init__(self):
        ABOD.__init__(self)

    def partial_fit(self, samples):
        pass

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


# ================================= #
#        Test Functions

def test_ABOD_1():

    clf = ABOD()

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_matthias_big.pkl")
    emb4 = load_embeddings("embeddings_laia.pkl")
    emb5 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # randomize data
    np.random.shuffle(emb2)

    # extract 99% variance subspace
    basis, mean = ExtractSubspace(emb2, 0.99)

    start = time.time()

    # reduce data
    data = ProjectOntoSubspace(emb2, mean, basis)
    dims = np.shape(data)

    # select minimum data to build convex hull
    min_nr_elems = dims[1]+4
    data_hull = data[0:min_nr_elems+1, :]

    print np.shape(data)

    # calculate hull
    #

    # ----------- Delauny tesselation
    if False:
        hull = Delaunay(data_hull)
        # print (hull.find_simplex(data[10, :]) >= 0)

        elems_in_hull = np.sum([1 if hull.find_simplex(sample) >= 0 else 0 for sample in data])
        print "Elements inside hull: {} | Hull points: {}".format(elems_in_hull, len(data_hull))

    # ----------- Convex hull (subgraph)
    if False:
        hull = ConvexHull(data_hull)
        # the vertices of the convex hull
        hull_points = hull.vertices
        # points inside hull
        print set(range(len(data_hull))).difference(hull.vertices)

    print "elements: {} | time: {}".format(min_nr_elems, time.time()-start)


def test_ABOD_2():

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_matthias_big.pkl")
    emb4 = load_embeddings("embeddings_laia.pkl")
    emb5 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # randomize data
    np.random.shuffle(emb2)

    # extract 99% variance subspace
    basis, mean = ExtractSubspace(emb2, 0.8)

    start = time.time()

    # reduce data
    data = ProjectOntoSubspace(emb2, mean, basis)
    dims = np.shape(data)

    # select minimum data to build convex hull
    min_nr_elems = dims[1] + 4
    data_hull = data[0:min_nr_elems + 1, :]

    print "Reduced data shape: {}".format(np.shape(data))

    # ----------- Delauny tesselation
    if True:
        print "Calculate hull using {} points".format(len(data_hull))
        hull = Delaunay(data_hull)
        # print (hull.find_simplex(data[10, :]) >= 0)

        test_data = data

        elems_in_hull = np.sum([1 if hull.find_simplex(sample) >= 0 else 0 for sample in test_data])
        print "Elements inside hull: {}/{} | Hull points: {}".format(elems_in_hull, len(test_data), len(data_hull))

    # ----------- Convex hull (subgraph)
    if False:
        hull = ConvexHull(data_hull)
        # the vertices of the convex hull
        hull_points = hull.vertices
        # points inside hull
        print set(range(len(data_hull))).difference(hull.vertices)

    print "elements: {} | time: {}".format(min_nr_elems, time.time() - start)


class OnlineHull():

    cluster = []

    def __init__(self):
        pass

    def init(self, samples):
        self.cluster = samples

    def partial_fit(self, samples):

        if len(self.cluster) == 0:
            print "NOT INITIALIZED YET! INITIALIZING USING INPUT SAMPLES"
            self.clusster = samples
            return

        # reduce cluster/sample data
        basis, mean = ExtractSubspace(self.cluster, 0.8)
        cluster_reduced = ProjectOntoSubspace(self.cluster, mean, basis)
        samples_reduced = ProjectOntoSubspace(samples, mean, basis)
        dims = np.shape(cluster_reduced)

        # select minimum data to build convex hull
        min_nr_elems = dims[1] + 4

        print "Recuding dimension: {}->{}".format(np.shape(self.cluster)[1], dims[1])

        data_hull = cluster_reduced     # take all samples of cluster
        print "Calculating cluster hull using {}/{} points".format(len(data_hull), len(self.cluster))
        hull = Delaunay(data_hull)

        print "Points included in hull: {}".format(np.unique(hull.convex_hull))

        # test original samples if inside convex hull
        elems_in_hull = np.sum([1 if hull.find_simplex(sample) >= 0 else 0 for sample in data_hull])
        print "Cluster elements INSIDE hull (to throw): {}/{}".format(elems_in_hull, len(data_hull))

        # include samples outside convex hull
        elems_in_hull = np.sum([1 if hull.find_simplex(sample) >= 0 else 0 for sample in samples_reduced])
        print "Elements INSIDE hull (to throw): {}/{}".format(elems_in_hull, len(samples))



        # reevaluate data (delete points inside hull)


def test_ABOD_3():
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_matthias_big.pkl")
    emb4 = load_embeddings("embeddings_laia.pkl")
    emb5 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # randomize data
    np.random.shuffle(emb2)

    cl = OnlineHull()

    # initialize with 10 samples
    cl.init(emb2[0:10,:])

    # partial update, check which samples are valuable
    cl.partial_fit(emb2[20:30,:])
    cl.partial_fit(emb2[30:40,:])

# ================================= #
#              Main

if __name__ == '__main__':
    test_ABOD_3()
