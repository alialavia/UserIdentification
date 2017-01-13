#!/usr/bin/env python2
import argparse
import os
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
import math
# analysis tools
from uids.utils.DataAnalysis import *
from sklearn import metrics
from sklearn.metrics.cluster import *
from external.jqmcvi.base import *
import time
import sys

from sklearn.metrics.pairwise import *
from sklearn.neighbors import DistanceMetric
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.neighbors.kde import KernelDensity

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory


def load_data(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def split_train_test(ds, training_percent=0.7, nr_training_batches=2):
    np.random.shuffle(ds)
    nr_training_samples = int(math.floor(training_percent*len(ds)))
    train = ds[0:nr_training_samples]
    test = ds[nr_training_samples:]

    if nr_training_batches == 1:
        return train, test

    while len(train) % nr_training_batches != 0:
        train = train[:-1]
    batch_size = len(train)/nr_training_batches
    train_split = np.array_split(train, nr_training_batches)

    return train_split, test


# ================================= #
#              Plotting


# ================================= #
#              Main

def test_1():

    # load embeddings
    emb_1 = load_data('embeddings_matthias.pkl')
    emb_2 = load_data('embeddings_elias.pkl')
    emb_3 = load_data('embeddings_laia.pkl')
    emb_lfw = load_data('embeddings_lfw.pkl')

    if emb_1 is None or emb_2 is None:
        print "--- embeddings could not be loaded. Aborting..."
        return

    train, test = split_train_test(emb_1, 0.7, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.9).fit(train)

    score = kde.score_samples(test)
    print score
    score = kde.score_samples(emb_2)
    print score


def test_2():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    score = kde.score_samples(X)
    print score

    plt.plot(score)
    plt.show()

if __name__ == '__main__':
    test_1()