#!/usr/bin/env python2
from sklearn.decomposition import PCA
from sklearn.utils.extmath import fast_dot
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import *

# ================================= #
#              Plotting

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


# ================================= #
#              Data Analysis

def ExtractMaxVarComponents(data, nr_components):
    """"""
    pca = PCA(n_components=np.size(data, 1))
    pca.fit(data)
    basis = pca.components_.T[:, 0:nr_components]
    expl_variance = np.sum(pca.explained_variance_ratio_[0:nr_components])
    return basis, pca.mean_, expl_variance

def ReduceToSubspace(data, explained_variance):
    basis, mean = ExtractSubspace(data, explained_variance)
    return ProjectOntoSubspace(data, mean, basis)

def ExtractSubspace(data, explained_variance):
    """"""
    pca = PCA(n_components=np.size(data, 1))
    pca.fit(data)
    var_listing = pca.explained_variance_ratio_

    s = 0
    index = 0
    for i, v in enumerate(var_listing):
        if s + v > explained_variance:
            break
        s = s + v
        index = i

    # extract basis
    basis = pca.components_.T[:, 0:(index+1)]
    return basis, pca.mean_


def ProjectOntoSubspace(data, mean, basis):
    """Applie dimension reduction"""
    reduced = data - mean
    reduced = fast_dot(reduced, basis)
    reduced = reduced + fast_dot(mean, basis)
    return reduced


def CalcComponentVariance(v):
    """Calculate Eigenbasis and the Eigenvector contributions"""
    pca = PCA(n_components=np.size(v,1))
    pca.fit(v)
    return pca.explained_variance_ratio_


def ExtractInverseSubspace(data, explained_variance):
    """ Extract subspace that explains 1-explained_variance % of the Variance

    Parameters
    ----------
    data : feature vectors
    explained_variance: percent of variance to crop
    """
    pca = PCA(n_components=np.size(data, 1))
    pca.fit(data)
    var_listing = pca.explained_variance_ratio_

    s = 0
    index = 0
    for i, v in enumerate(var_listing):
        if s + v > explained_variance:
            break
        s = s + v
        index = i

    # extract basis inverse
    basis = pca.components_.T[:, (index+1):np.size(data, 1)]
    return basis, pca.mean_

# ================================= #
#   Cluster Validity Indices (CVI)


def silhouette_index(cluster_list, metric='euclidean'):

    # The Silhouette Index measure the distance between each data point, the centroid of the cluster
    # it was assigned to and the closest centroid belonging to another cluster
    # - normalized, close to 1 = good

    # generate labels
    labels = []
    data = []
    for i in range(len(cluster_list)):
        labels = np.concatenate((labels, np.repeat(i, np.size(cluster_list[i],0))))
        if i == 0:
            data = cluster_list[i]
        else:
            np.concatenate((data, cluster_list[i]), axis=0)
    return silhouette_score(data, labels, metric=metric)


def dunn_index(cluster_list, metric='euclidean'):
    """ Dunn index
    - the higher the better
    - normalized
    - not robust - instable to outliers!

    Parameters
    ----------
    cluster_list : list of np.arrays, each containing feature vectors containing to the same cluster
    """

    deltas = np.ones([len(cluster_list), len(cluster_list)]) * 1000000
    big_deltas = np.zeros([len(cluster_list), 1])
    l_range = list(range(0, len(cluster_list)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = np.min(pairwise_distances(cluster_list[k], cluster_list[l], metric))

        big_deltas[k] = np.max(pairwise_distances(cluster_list[k], metric=metric))

    print "--- up: {}, down: {}".format(np.min(deltas), np.max(big_deltas))

    di = np.min(deltas) / np.max(big_deltas)
    return di


def daviesbouldin_index(cluster_list, cluster_centers, metric='euclidean'):
    """ Davis Bouldin Index
    - the smaller the better
    - Davies-Bouldin Index evaluates intra-cluster similarity and inter-cluster differences
    - not normalized, difficult to compare 2 values from different data sets

    Parameters
    ----------
    cluster_list : list of np.arrays, each containing feature vectors containing to the same cluster
    cluster_centers : np.array, center of the clusters, same order as cluster_list
    """

    nr_clusters = len(cluster_list)
    rel_intra_class_sep = np.zeros([nr_clusters], dtype=np.float64)
    db = 0

    for k in range(nr_clusters):
        rel_intra_class_sep[k] = np.sum(pairwise_distances(cluster_list[k], cluster_centers[k], metric=metric)) / len(cluster_list[k])

    c_separations = pairwise_distances(cluster_centers, metric=metric)

    for k in range(nr_clusters):
        values = np.zeros([nr_clusters - 1], dtype=np.float64)
        for l in range(0, k):
            values[l] = (rel_intra_class_sep[k] + rel_intra_class_sep[l]) / c_separations[k, l]
        for l in range(k + 1, nr_clusters):
            values[l - 1] = (rel_intra_class_sep[k] + rel_intra_class_sep[l]) / c_separations[k, l]

        db += np.max(values)
    res = db / nr_clusters
    return res
