import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import pickle
import os
from sklearn import svm
from sklearn import linear_model
from sklearn.decomposition import PCA
from uids.utils.DataAnalysis import *

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return embeddings
    return None


def compare_principle_components(data, reference_ds, nr_comps):

    pca = PCA(n_components=128)
    pca.fit(reference_ds)

    ref_variance = pca.explained_variance_ratio_[0:nr_comps]
    ref_basis = pca.components_.T[:, 0:nr_comps]
    ref_mean = np.mean(reference_ds, axis=0)

    # --------------- calculations

    basis_vectors = [None]*len(data)
    variance = [None]*len(data)

    for i_ds, ds in enumerate(data):
        pca = PCA(n_components=128)
        pca.fit(ds)
        basis_vectors[i_ds] = pca.components_.T[:, 0:nr_comps]
        variance[i_ds] = pca.explained_variance_ratio_[0:nr_comps]

    # ----------------  plot variance components
    plt.figure()
    plt.title("Variance Ratio of Principle Components")
    for i_ds, var in enumerate(variance):
        # calculate distance between vectors (to general ds)
        x = range(0, nr_comps)
        plt.plot(x, variance[i_ds])
    plt.xlabel("Component Number")
    plt.ylabel("Explained Variance")

    # ----------------  plot basis
    plt.figure()
    plt.title("Basis vector differences")
    for i_ds, basis_i in enumerate(basis_vectors):
        y = [None] * nr_comps
        for i in range(0, nr_comps):
            a = pairwise_distances(ref_basis[:, i].reshape(1, -1), basis_i[:,i].reshape(1, -1), metric='euclidean')
            y[i] = a[0,0]
        # print vector_i
        x = range(0, nr_comps)
        plt.plot(x, y)
    plt.xlabel("Component Number")
    plt.ylabel("Euclidean Distance")

    # ----------------  variance reduction in general basis


    # # project onto
    # for i_ds, ds in enumerate(data):
    #
    #
    # # project onto general basis
    # var_orig = np.var(emb1)
    #
    #
    #
    # ProjectOntoSubspace(data, ref_mean, ref_basis)
    #
    #
    # reduced = data - mean
    # reduced = fast_dot(reduced, basis)
    # reduced = reduced + fast_dot(mean, basis)
    #
    # print np.var(emb1)
    # print np.var(emb2)

    plt.show()





# ================================= #
#              Tests

def test1():
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_christian.pkl")
    emb4 = load_embeddings("embeddings_laia.pkl")
    emb5 = load_embeddings("embeddings_matthias_big.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")


    data = [emb1, emb2, emb4, emb5]


    compare_principle_components(data, emb_lfw, 20)




    # s = 0
    # index = 0
    # for i, v in enumerate(var_listing):
    #     if s + v > explained_variance:
    #         break
    #     s = s + v
    #     index = i
    #
    # # extract basis
    # basis = pca.components_.T[:, 0:(index+1)]
    # return basis, pca.mean_



# ================================= #
#              Main

if __name__ == '__main__':

    test1()

