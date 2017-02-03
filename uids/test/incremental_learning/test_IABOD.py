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
from uids.utils.Logger import Logger as log
from sklearn.neighbors import NearestNeighbors
from uids.utils.lof import LocalOutlierFactor
from uids.utils.KNFilter import KNFilter
from uids.online_learning.IABOD import IABOD


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


def test_ABOD_3():
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_matthias_big.pkl")
    emb4 = load_embeddings("embeddings_laia.pkl")
    emb5 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # randomize data
    np.random.shuffle(emb3)
    np.random.shuffle(emb2)

    cl = IABOD()

    # initialize with 10 samples
    cl.fit(emb2[0:10,:])


    # partial update from different scene, check which samples are valuable
    training_data = emb3[0:1000]
    test_data = emb3[1000:1100]
    test_data2 = emb5[0:100]
    batch_size = 10
    nr_steps = int(len(training_data/batch_size))
    nr_steps = 200

    cl_errors = []
    cl_unknowns = []
    cl_errors2 = []
    cl_unknowns2 = []

    for step in range(0, nr_steps):
        start = (batch_size*step)
        stop = start + batch_size


        # prediction = cl.predict(test_data)
        # cl_errors.append(len(prediction[prediction < 0]))
        # cl_unknowns.append(len(prediction[prediction == 0]))
        # # test outlier
        # prediction = cl.predict(test_data2)
        # cl_errors2.append(len(prediction[prediction > 0]))
        # cl_unknowns2.append(len(prediction[prediction == 0]))


        cl.partial_fit(training_data[start:stop, :])


    # plt.figure()
    # plt.plot(range(0, nr_steps), cl_errors, label="Error1")
    # plt.plot(range(0, nr_steps), cl_unknowns, label="Unkowns1")
    # plt.plot(range(0, nr_steps), cl_errors2, label="Error2")
    # plt.plot(range(0, nr_steps), cl_unknowns2, label="Unkowns2")
    # plt.title("Classification Error")
    # plt.xlabel("Iteration")
    # plt.ylabel("Error [nr samples]")
    # plt.legend()

    # plot classification performance

    cl.disp_log()

# ================================= #
#              Main

if __name__ == '__main__':
    test_ABOD_3()
