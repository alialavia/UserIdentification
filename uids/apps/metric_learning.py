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
from metric_learn import Covariance
import time
from uids.online_learning.ABOD import ABOD

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models', 'embedding_samples')	# path to the model directory

from metric_learn import *
from sklearn.datasets import load_iris

def load_data(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


# ================================= #
#              Main

# Metrics:
# ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
# 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

def evaluate_metric(c1_samples, c2_samples, metric='cosine'):
    # calc centroid of fitting data
    mean = np.mean(c1_samples, axis=0)
    # std = np.std(c1_samples, axis=0)

    # inter-class centroid separation distribution
    intercl_dist = pairwise_distances(mean, c1_samples, metric=metric)
    max_intercl = intercl_dist.max(axis=1)
    min_intercl = intercl_dist.min(axis=1)

    # neighbour separation distribution
    out_dist = pairwise_distances(mean, c2_samples, metric=metric)
    max_out = out_dist.max(axis=1)
    min_out = out_dist.min(axis=1)

    print "--- min: {}, max: {}".format(min_intercl, max_intercl)

    # plot bins
    n, bins, patches = plt.hist(np.transpose(intercl_dist), 50, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(np.transpose(out_dist), 50, normed=1, facecolor='red', alpha=0.75)

    # plot title, axis,
    plt.title('Metric evaluation: {}-distance'.format(metric))
    plt.ylabel('Number of samples')
    plt.xlabel('Distance from sample to class centroid')



def test_lmnn_sample_data():

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']
    d0 = X[Y == 0]
    d1 = X[Y == 1]
    lmnn = LMNN(k=5, learn_rate=1e-6)
    lmnn.fit(X, Y)

    metric_transformer = lmnn.transformer()

    print np.shape(metric_transformer)
    print np.shape(d0)
    d0_new = np.dot(d0, np.transpose(metric_transformer))
    d1_new = np.dot(d1, np.transpose(metric_transformer))

    evaluate_metric(d0, d1, 'euclidean')
    plt.show()
    evaluate_metric(d0_new, d1_new, 'euclidean')
    plt.show()


def test_lmnn(d0, d1, learner):
    # generate input
    labels = np.concatenate((np.repeat(0, len(d0)), np.repeat(1, len(d1))))
    data = np.concatenate((d0, d1))

    learner.fit(data, labels)
    metric_transformer = learner.transformer()
    d0_new = np.dot(d0, np.transpose(metric_transformer))
    d1_new = np.dot(d1, np.transpose(metric_transformer))

    evaluate_metric(d0, d1, 'euclidean')
    plt.show()
    evaluate_metric(d0_new, d1_new, 'euclidean')
    plt.show()


def test_lmnn0():

    d0 = load_data('embeddings_matthias.pkl')
    d1 = load_data('embeddings_christian.pkl')

    d0_train = d0[0:30,:]
    d1_train = d1[0:30,:]
    d0_test = d0[30:40,:]
    d1_test = d1[30:40,:]

    # ---------- train
    labels_train = np.concatenate((np.repeat(0, len(d0_train)), np.repeat(1, len(d1_train))))
    data_train = np.concatenate((d0_train, d1_train))

    lmnn = LMNN(k=3, learn_rate=1e-6)
    start = time.time()
    lmnn.fit(data_train, labels_train)
    print "Fitting took {} seconds".format(time.time()-start)

    # ---------- test

    print "---- Evaluation in original space: Metric against Class 0"
    print "     Smaller valuer = better choice"
    cos_dist_orig00 = np.mean(pairwise_distances(d0_test, d0_train, metric='cosine'))
    cos_dist_orig01 = np.mean(pairwise_distances(d0_test, d1_train, metric='cosine'))
    cos_dist_orig10 = np.mean(pairwise_distances(d1_test, d0_train, metric='cosine'))
    cos_dist_orig11 = np.mean(pairwise_distances(d1_test, d1_train, metric='cosine'))
    print "Class 0 samples: Cosine distance: 0 - {:2f}, 1 - {:2f}".format(cos_dist_orig00, cos_dist_orig01)
    print "Class 1 samples: Cosine distance: 0 - {:2f}, 1 - {:2f}".format(cos_dist_orig10, cos_dist_orig11)

    print "---- Evaluation in learned space:"
    print "     Smaller valuer = better choice"
    cos_dist_orig00 = np.mean(pairwise_distances(lmnn.transform(d0_test), lmnn.transform(d0_train), metric='cosine'))
    cos_dist_orig01 = np.mean(pairwise_distances(lmnn.transform(d0_test), lmnn.transform(d1_train), metric='cosine'))
    cos_dist_orig10 = np.mean(pairwise_distances(lmnn.transform(d1_test), lmnn.transform(d0_train), metric='cosine'))
    cos_dist_orig11 = np.mean(pairwise_distances(lmnn.transform(d1_test), lmnn.transform(d1_train), metric='cosine'))
    print "Class 0 samples: Cosine distance: 0 - {:2f}, 1 - {:2f}".format(cos_dist_orig00, cos_dist_orig01)
    print "Class 1 samples: Cosine distance: 0 - {:2f}, 1 - {:2f}".format(cos_dist_orig10, cos_dist_orig11)


    print "===========================ABOD===================================="

    clf0_orig = ABOD()
    clf1_orig = ABOD()
    clf0_opt = ABOD()
    clf1_opt = ABOD()

    # fit classifiers
    clf0_orig.fit(d0_train)
    clf1_orig.fit(d1_train)
    clf0_opt.fit(lmnn.transform(d0_train))
    clf1_opt.fit(lmnn.transform(d1_train))

    # predict
    print "\n-----------ABOD values in original space:------------------\n\n"
    clf0_orig.predict(d0_test)
    clf0_orig.predict(d1_test)
    clf1_orig.predict(d0_test)
    clf1_orig.predict(d1_test)

    # predict
    print "\n-----------ABOD values in custom space:------------------\n\n"
    clf0_opt.predict(lmnn.transform(d0_test))
    clf0_opt.predict(lmnn.transform(d1_test))
    clf1_opt.predict(lmnn.transform(d0_test))
    clf1_opt.predict(lmnn.transform(d1_test))


def test_covariance():
    iris = load_iris()['data']

    cov = Covariance().fit(iris)
    x = cov.transform(iris)
    print x


def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Image folder.", default="faces")
    parser.add_argument('--output', help="Statistics output folder.", default="stats")
    args = parser.parse_args()


    # load embeddings
    emb_0 = load_data('embeddings_matthias.pkl')
    emb_1 = load_data('embeddings_elias.pkl')
    emb_2 = load_data('embeddings_laia.pkl')
    emb_3 = load_data('embeddings_christian.pkl')
    emb_lfw = load_data('embeddings_lfw.pkl')

    if emb_0 is None or emb_1 is None:
        print "--- embeddings could not be loaded. Aborting..."
        return

    # ------------------- EVALUATION

    d0 = emb_0
    d1 = emb_3

    # learner = LMNN(k=2, learn_rate=1e-6)
    # learner = NCA(max_iter=100, learning_rate=0.01)
    # learner = LFDA(k=2, dim=100)
    # learner = RCA_Supervised(num_chunks=30, chunk_size=2)
    learner = LSML_Supervised(num_constraints=400)

    # ---------------------------




    test_lmnn(emb_0, emb_3, learner)

    return

    labels = np.concatenate((np.repeat(0, len(d0)), np.repeat(1, len(d1))))
    data = np.concatenate((d0, d1))

    learner.fit(data, labels)
    metric_transformer = learner.transformer()
    d0_new = np.dot(d0, np.transpose(metric_transformer))
    d1_new = np.dot(d1, np.transpose(metric_transformer))

    evaluate_metric(d0, d1, 'cosine')
    plt.show()
    evaluate_metric(d0_new, d1_new, 'cosine')
    plt.show()

    # test transformation on whole dataset: does everybody split?
    evaluate_metric(d0_new, np.dot(emb_lfw, np.transpose(metric_transformer)), 'euclidean')
    plt.show()


    # ------ metric learning against general ds on subspace (memory issues)

    # basis, mean = ExtractSubspace(emb_lfw, 0.99)
    # emb_0_r = ProjectOntoSubspace(emb_0, mean, basis)
    # emb_lfw_r = ProjectOntoSubspace(emb_lfw, mean, basis)
    #
    # labels = np.concatenate((np.repeat(0, len(emb_0_r)), np.repeat(1, len(emb_lfw_r))))
    # data = np.concatenate((emb_0_r, emb_lfw_r))
    #
    # learner.fit(data, labels)
    # metric_transformer = learner.transformer()
    # emb_0_r = np.dot(emb_0_r, np.transpose(metric_transformer))
    # emb_lfw_r = np.dot(emb_lfw_r, np.transpose(metric_transformer))
    #
    # # plot_separation_dist(emb_0_r)
    # plot_ds_separations(emb_0_r, emb_lfw_r)



if __name__ == '__main__':
    test_lmnn0()