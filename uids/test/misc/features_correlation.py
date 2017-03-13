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
import seaborn as sns


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


def eval_correlation():
    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_matthias_big.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb4 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    ref = emb1[0:40,:]
    test = emb1[40:60,:]
    ul = emb4[0:10,:]

    # plt.plot(np.arange(0,128), ul[0])
    # # plt.plot(np.arange(0,128), ul[1])
    # # plt.plot(np.arange(0,128), ul[2])
    #
    #
    # for i in range(0,10):
    #     plt.plot(np.arange(0, 128), ul[i], color='blue')
    # plt.plot(np.arange(0,128), test[3], color="red")
    # plt.show()


    # sns.set(context="paper", font="monospace")
    #
    # # Load the datset of correlations between cortical brain networks
    # df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
    #
    #
    # corrmat = df.corr()
    #
    # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(12, 9))
    #
    # # Draw the heatmap using seaborn
    # sns.heatmap(corrmat, vmax=.8, square=True)
    #
    # # Use matplotlib directly to emphasize known networks
    # networks = corrmat.columns.get_level_values("network")
    # for i, network in enumerate(networks):
    #     if i and network != networks[i - 1]:
    #         ax.axhline(len(networks) - i, c="w")
    #         ax.axvline(i, c="w")
    # f.tight_layout()
    # plt.show()


    print emb1[0,:]

    # plotting the correlation matrix
    R = np.corrcoef(emb1[0,:])
    plt.pcolor(R)
    plt.colorbar()
    plt.yticks(np.arange(0.5, 10.5), range(0, 10))
    plt.xticks(np.arange(0.5, 10.5), range(0, 10))
    plt.show()


# ================================= #
#              Main

if __name__ == '__main__':
    eval_correlation()


