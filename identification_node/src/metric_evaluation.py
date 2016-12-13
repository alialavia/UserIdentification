#!/usr/bin/env python2
import argparse
import os
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
import math
# analysis tools
from lib.DataAnalysis import *
from sklearn import metrics
from sklearn.metrics.cluster import *
from external.jqmcvi.base import *
import time

from sklearn.metrics.pairwise import *

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


# ================================= #
#              Plotting

# Metrics:
# ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
# 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

def plot_dist_to_centroid(c1_samples, c2_samples, metric='cosine'):
    # calc centroid of fitting data
    mean = np.mean(c1_samples, axis=0)
    # std = np.std(c1_samples, axis=0)
    print "---metric: "+metric
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

    # add cluster indices
    k_list = [c1_samples, c2_samples]
    k_centers = [np.mean(c1_samples, axis=0), np.mean(c2_samples, axis=0)]

    # dunn = dunn_index(k_list, metric)
    # daviesbouldin = daviesbouldin_index(k_list, k_centers, metric)
    # plt.text(0.4, 5, "dunn: {}, db: {}".format(dunn, daviesbouldin), va='top', fontsize=9)

    # plot title, axis,
    plt.title('Metric evaluation: {}-distance'.format(metric))
    plt.ylabel('Number of samples')
    plt.xlabel('Distance from sample to class centroid')



# ================================= #
#        Evaluation Routines


def plot_ds_separations(ds1, ds2, metric='euclidean'):

    # Cosine distance is defined as 1.0 minus the cosine similarity.

    if metric == 'cosine_similarity':
        sep = cosine_similarity(ds1, ds2)
    else:
        sep = pairwise_distances(ds1, ds2, metric=metric)

    max_out = np.amax(sep)
    min_out = np.amin(sep)
    print "--- min: {}, max: {}".format(min_out, max_out)

    fig = plt.figure()
    n, bins, patches = plt.hist(np.transpose(sep), 50, normed=1, facecolor='green', alpha=0.75)
    plt.title('Inter-Class separation: {}-distance'.format(metric))
    plt.ylabel('Number of samples')
    plt.xlabel('Sample separation')
    plt.show()

def plot_separation_dist(ds1, metric='euclidean'):

    print "=====================================================\n"
    print "  INTRA-CLASS SEPARATION IN {} DISTANCE\n".format(metric)
    print "=====================================================\n"

    if metric == 'cosine_similarity':
        sep = cosine_similarity(ds1)
    else:
        sep = pairwise_distances(ds1, metric=metric)

    fig = plt.figure()
    n, bins, patches = plt.hist(np.transpose(sep), 50, normed=1, facecolor='green', alpha=0.75)
    plt.title('Intra-Class separation: {}-distance'.format(metric))
    plt.ylabel('Number of samples')
    plt.xlabel('Sample separation')

    # extract 99.9% subspace
    basis, mean = ExtractInverseSubspace(ds1, 1 - 0.1)
    print "--- reduced dimension to: {}".format(np.size(basis, 1))

    # project data onto subspace
    ds1_sub = ProjectOntoSubspace(ds1, mean, basis)

    # add mean

    if metric == 'cosine_similarity':
        sep = cosine_similarity(ds1)
    else:
        sep = pairwise_distances(ds1, metric=metric)


    fig = plt.figure()
    n, bins, patches = plt.hist(np.transpose(ds1_sub), 50, normed=1, facecolor='green', alpha=0.75)
    plt.title('Intra-Class separation on 0.2 Var subspace: {}-distance'.format(metric))
    plt.ylabel('Number of samples')
    plt.xlabel('Sample separation')
    plt.show()


def print_distr_on_minvar_subspace(reference_ds, general_ds, d_eval_variance, explained_variance = 0.2, metric='euclidean'):

    print "=====================================================\n"
    print "  EVALUATE DISTRIBUTION MINIMAL VARIANCE ({}) SUBSPACE\n".format(explained_variance)
    print "=====================================================\n"

    # extract 99.9% subspace
    basis, mean = ExtractInverseSubspace(d_eval_variance, 1-explained_variance)
    print "--- reduced dimension to: {}".format(np.size(basis,1))

    # project data onto subspace
    d1_proj = ProjectOntoSubspace(reference_ds, mean, basis)
    d2_proj = ProjectOntoSubspace(general_ds, mean, basis)
    plot_dist_to_centroid(d1_proj, d2_proj, metric)

    k_list = []
    k_list.append(d1_proj)
    k_list.append(d2_proj)
    k_centers = []
    k_centers.append(np.mean(d1_proj, axis=0))
    k_centers.append(np.mean(d2_proj, axis=0))
    try:
        print "--- Dunn Index index: {}".format(dunn_index(k_list, metric))
        print "--- Davies Bouldin Index: {}".format(daviesbouldin_index(k_list, k_centers, metric))
    except MemoryError:
        pass
    print "=====================================================\n"
    plt.show()


def print_distr_on_subspace(reference_ds, general_ds, d_eval_variance, metric='euclidean'):

    print "=====================================================\n"
    print "        EVALUATE DISTRIBUTION ON 99.9% SUBSPACE\n"
    print "=====================================================\n"

    # extract 99.9% subspace
    basis, mean = ExtractSubspace(d_eval_variance, 0.999)
    print "--- reduced dimension to: {}".format(np.size(basis,1))

    # project data onto subspace
    d1_proj = ProjectOntoSubspace(reference_ds, mean, basis)
    d2_proj = ProjectOntoSubspace(general_ds, mean, basis)
    plot_dist_to_centroid(d1_proj, d2_proj, metric)

    k_list = []
    k_list.append(d1_proj)
    k_list.append(d2_proj)
    k_centers = []
    k_centers.append(np.mean(d1_proj, axis=0))
    k_centers.append(np.mean(d2_proj, axis=0))
    try:
        print "--- Dunn Index index: {}".format(dunn_index(k_list, metric))
        print "--- Davies Bouldin Index: {}".format(daviesbouldin_index(k_list, k_centers, metric))
    except MemoryError:
        pass
    print "=====================================================\n"
    plt.show()

def plot_different_distance_metrics(main_distr, general_distr):

    # start plotting
    fig = plt.figure()
    # min 128 samples: , 'mahalanobis'
    # metrics = ['euclidean', 'cosine', 'braycurtis', 'correlation', 'minkowski', 'seuclidean', 'sqeuclidean']
    metrics = ['euclidean', 'cosine']

    cols = 2.0
    rows = int(math.ceil(len(metrics) / cols))
    plot_format = str(rows)+str(int(cols))

    for i, m in enumerate(metrics):
        code = plot_format+str(i+1)
        try:
            ax = fig.add_subplot(int(code))
            plot_dist_to_centroid(main_distr, general_distr, m)
        except MemoryError:
            print "--- Out of memory: Could not calculate cluster metrics for {} metric".format(m)
        except Exception:
            print "--- Could not plot {} metric".format(m)
            continue

    plt.grid(True)
    plt.show()

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

    # 1.1 Plot different distance metrics
    plot_different_distance_metrics(emb_1, emb_2)

    # 1.2 Plot distance to centroid on 99.9% general subspace
    print_distr_on_subspace(emb_1, emb_2, emb_lfw, 'cosine')

    # 2. INTRA-/INTER-CLASS DISTANCE DISTRIBUTION
    plot_separation_dist(emb_1, 'cosine')
    plot_ds_separations(emb_1, emb_lfw, 'cosine')

if __name__ == '__main__':
    run_evaluation()