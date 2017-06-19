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
from PIL import Image

from sklearn.metrics.pairwise import *
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
import numpy as np
from scipy import stats
from scipy import misc

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory
ressourceDir = os.path.join(fileDir, '..', 'ressource')	# path to the model directory


def load_data(filename):
    filename = "{}/{}".format(modelDir, filename)

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def plot_component_variance(data, label=""):
    pca = PCA(n_components=128)
    pca.fit(data)
    var_listing = pca.explained_variance_ratio_

    # build sum
    s = []
    index = 0
    for i, v in enumerate(var_listing):

        if i==0:
            s.append(v)
        else:
            s.append(s[i-1]+v)

    plt.plot(np.arange(0,128), s)

    print "Data: ", list(s)


# ================================= #
#              Plotting


if __name__ == '__main__':

    emb_1 = load_data('embeddings_lfw.pkl')
    emb_2 = load_data('matthias/pose_matthias3.pkl')
    emb_3 = load_data('matthias/pose_matthias2.pkl')

    plot_component_variance(emb_1)
    plot_component_variance(emb_2)
    # plot_component_variance(emb_3)
    plt.show()