
import argparse
import os

import numpy as np
import time
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.spatial.distance import *

from sklearn.utils.extmath import fast_dot
from sklearn.utils import check_array
import sys

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))


def load_embeddings(self, filename):
    filename = "{}/svm_classifier.pkl".format(fileDir)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return embeddings
    return None

"""
metric: 'euclidean',
"""
def calc_dist_from_mean(embeddings, mean, metric):
    distances = pairwise_distances(mean, embeddings, metric='euclidean')
    max_dist = distances.max(axis=1)
    min_dist = distances.min(axis=1)
    return min_dist, max_dist

def eval_in_stddev(embeddings, mean, std):

    total = len(embeddings)
    for emb in embeddings:
        continue

def plot_histo(x, title='Distances'):
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    plt.title(title)
    plt.grid(True)

    plt.show()


def ICPfit(embeddings):
    ipca = IncrementalPCA(n_components=2, batch_size=3)
    ipca.fit(embeddings_1)
    IncrementalPCA(batch_size=3, copy=True, n_components=2, whiten=False)
    reduced = ipca.transform(embeddings)
    return reduced

def PCAcosineDistance(v, v_mean):
    pca = PCA(n_components=50)
    pca.fit(embeddings_1)
    v_t = pca.transform(v)
    v_t_mean = pca.transform(v_mean)
    return cosine(v_t, v_t_mean)

def GetEigenVectors(v):
    pca = PCA(n_components=128)
    pca.fit(v)
    print len(pca.components_)
    print pca.noise_variance_

def PCAreduced(input, nr_comps = 20):
    if nr_comps > np.size(input,0):
        raise ValueError('PCA failed')
    pca = PCA(n_components=nr_comps)
    pca.fit(input)
    out = pca.transform(input)
    return out

def PCAreducedInverse(input, nr_comps = 20):
    dims = np.size(input, 1)
    if nr_comps > np.size(input, 0):
        raise ValueError('PCA failed')
    pca = PCA(n_components=dims)
    pca.fit(input)

    # --------- extract last basis vectors
    nr_basis_vectors = np.size(pca.components_.T, 1)
    start_index = dims-nr_comps
    basis = pca.components_.T[:,start_index:dims]

    input = input - pca.mean_
    out = fast_dot(input, basis)
    return out


def ProjectMinVariance(input_vectors, core_embeddings, dims, base_components):

    print "PCA data: {} samples, {} dim".format(np.size(input_vectors,0), np.size(input_vectors,1))

    # calculate principal components
    pca = PCA(n_components=20)
    pca.fit(core_embeddings)

    if pca.mean_ is None:
        raise ValueError('PCA failed')

    input_vectors = [input_vectors[0,:]]

    print np.shape(input_vectors)

    print "Input data: {} samples, {} dim".format(np.size(input_vectors,0), np.size(input_vectors,1))
    transformed = pca.transform(input_vectors)
    print np.shape(transformed)


    return
    # print np.size(pca.components_.T,0)
    # print np.size(pca.components_.T,1)

    # -----------------------------------------------------------------
    # extract last basis vectors
    # nr_elems = dims
    # nr_basis_vectors = np.size(pca.components_.T, 1)
    # start_index = nr_basis_vectors-nr_elems
    # basis = pca.components_.T[:,start_index:nr_basis_vectors]

    # -----------------------------------------------------------------

    # print np.size(basis, 0)
    # print np.size(basis, 1)

    projected = []

    # for i, v in enumerate(to_reduce):

    #     proj = pca.transform(v[0])
    #     # print np.shape(proj)
    # projected.append(proj)

    # print np.size(pca.components_.T,0)
    # print np.size(pca.components_.T,1)
    # print pca.explained_variance_ratio_




    # # zero mean
    for i, v in enumerate(to_reduce):

        print(np.shape(v))

        v_projected = pca.transform(v)

        print v_projected
        # vec = check_array(v)
        # print np.shape(pca.mean_)
        # print np.shape(v)
        # print np.shape(vec)
        # return
        # vec = vec - pca.mean_
        # v_projected = fast_dot(vec, basis)
        # # project


        projected.append(v_projected)


    return projected

def eval_dist_from_mean(vectors, metric='cosine'):

    # mean deviation
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0)

    # calculate vector distances to mean
    distances = pairwise_distances(mean, vectors, metric=metric)
    max_dist = distances.max(axis=1)
    min_dist = distances.min(axis=1)

    # plot
    plot_histo(np.transpose(distances), 'Distance to mean ['+metric+']')



# ================================= #
#              Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Image folder.", default="faces")
    parser.add_argument('--output', help="Statistics output folder.", default="stats")

    # parse arguments
    args = parser.parse_args()

    filename = "embeddings_christian.pkl"

    embeddings_1 = None
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings_1 = pickle.load(f)
            f.close()

    embeddings_1 = np.array(embeddings_1)


    print np.shape(embeddings_1)
    # limit
    #embeddings_1 = embeddings_1[1:10,:]


    if embeddings_1 is not None:
        projected = PCAreducedInverse(embeddings_1, 100)
        eval_dist_from_mean(projected, 'euclidean')
        # project onto min variance components
        # projected = ProjectMinVariance(embeddings_1, embeddings_1, 2, 128)

        # print "Dimensions: before {}, after: {}".format(np.size(embeddings_1, 1), np.size(projected, 1))

        # print np.shape(projected)

        # print(np.size(embeddings_1,1))
        # print(np.size(embeddings_1,0))
        #
        #
        #
        # eval_dist_from_mean(projected, 'euclidean')
        #

