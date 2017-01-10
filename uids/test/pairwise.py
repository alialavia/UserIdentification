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

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models', 'embedding_samples')	# path to the model directory

def load_data(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def benchmark_pairwise(samples, metric='cosine'):

    start = time.time()
    # calc centroid of fitting data
    mean = np.mean(samples, axis=0)
    # std = np.std(c1_samples, axis=0)
    # inter-class centroid separation distribution
    intercl_dist = pairwise_distances(mean, samples, metric=metric)
    print "--- took: {}s".format(time.time()-start)


# ================================= #
#              Plotting


# ================================= #
#              Main

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Image folder.", default="faces")
    parser.add_argument('--output', help="Statistics output folder.", default="stats")
    args = parser.parse_args()

    # load embeddings
    emb_1 = load_data('embeddings_matthias.pkl')
    emb_2 = load_data('embeddings_elias.pkl')
    emb_3 = load_data('embeddings_laia.pkl')
    emb_lfw = load_data('embeddings_lfw.pkl')

    if emb_1 is None or emb_2 is None:
        print "--- embeddings could not be loaded. Aborting..."
        return

    # ------------------- START EVALUATION

    # 1. DISTANCE TO CENTROID
    print(len(emb_lfw))
    benchmark_pairwise(emb_lfw)

    start = time.time()
    mahal = DistanceMetric.get_metric('mahalanobis', V=np.cov(emb_1))

    # fit a Minimum Covariance Determinant (MCD) robust estimator to data

    # fit a Minimum Covariance Determinant (MCD) robust estimator to data
    robust_cov = MinCovDet().fit(emb_1)

    # compare estimators learnt from the full data set with true parameters
    emp_cov = EmpiricalCovariance().fit(emb_1)

    # mahal_emp_cov = emp_cov.mahalanobis(emb_1[0])
    # mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)

    print "--- took: {}s".format(time.time()-start)
    print len(emb_1[0])


if __name__ == '__main__':
    run_evaluation()