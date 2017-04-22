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
ressourceDir = os.path.join(fileDir, '../..', 'ressource')	# path to the model directory


def load_data(filename):
    filename = "{}/{}".format(modelDir, filename)

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def display_image(embedding_name, indices, img_folder_name=""):

    filename = "{}/{}_image_names.pkl".format(modelDir, embedding_name)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            picture_names = pickle.load(f)
            f.close()
            if img_folder_name == "":
                img_folder_name = embedding_name


            size = 250
            images = []
            widths = []
            heights = []

            # load images
            for i in indices:
                img_name = "{}/{}/{}".format(ressourceDir, img_folder_name, picture_names[i])
                image = misc.imread(img_name)
                height, width, dims = image.shape
                image = misc.imresize(image, (size, size))
                images.append(image)
                widths.append(width)
                heights.append(height)

            new_im = np.hstack(images)
            print new_im.shape


            plt.imshow(new_im, aspect="auto")
            plt.show()

            # misc.pilutil.imshow(image)


def load_labels(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        my_data = genfromtxt(filename, delimiter=',')
        return my_data
    return None

# ---------------------------------------------

def plot_pitch_yaw_comparison():

    emb_pitch = load_data("embeddings_pitch.pkl")
    emb_yaw = load_data("embeddings_yaw.pkl")

    metric = 'euclidean'
    emb_pitch = emb_pitch[::-1]

    # select range
    # emb_yaw = emb_yaw[0:-2,:]
    # emb_pitch = emb_pitch[1:,:]

    # emb_pitch = emb_pitch[0:5,:]

    sep = pairwise_distances(emb_pitch, emb_yaw, metric=metric)
    if metric=='euclidean':
        sep = np.square(sep)

    plt.imshow(sep, cmap='GnBu_r', interpolation='nearest')
    cbar = plt.colorbar()
    # cbar = pl.colorbar(G, ticks=range(g1, g2 + 1))
    # cbar.ax.set_ylabel('Gradient (%)', fontsize=10)

    cl = plt.getp(cbar.ax, 'ymajorticklabels')
    plt.setp(cl, fontsize=16)

    if metric == 'cosine':
        cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    else:
        cbar.set_ticks([0.3, 0.5, 0.7, 0.9, 1.1])

    # cbar.set_ticklabels([mn, md, mx])

    plt.xticks([0, 3, 6, 9, 12])
    plt.xlim([-0.5, 12.5])
    plt.yticks([0, 3, 6, 9, 12])
    plt.ylim([-0.5, 12.5])
    plt.show()


def plot_pitch_yaw_gain():

    emb_pitch = load_data("embeddings_pitch.pkl")
    emb_yaw = load_data("embeddings_yaw.pkl")

    metric = 'euclidean'

    emb_pitch = emb_pitch[::-1]

    # select range
    # emb_yaw = emb_yaw[0:-2,:]
    # emb_pitch = emb_pitch[1:,:]

    center = (len(emb_pitch)+1)/2 - 1
    sep = pairwise_distances(emb_pitch[center], emb_pitch, metric='euclidean')
    sep = np.square(sep)
    print list(sep[0])


    center = (len(emb_yaw) + 1) / 2 - 1
    sep = pairwise_distances(emb_yaw[center], emb_yaw, metric='euclidean')
    sep = np.square(sep)
    print list(sep[0])


def calc_nearest_pose():

    emb = load_data("compare_nearest_pose.pkl")
    emb2 = load_data("christian_clean.pkl")
    emb3 = load_data("matthias_test.pkl")


    sep = pairwise_distances(emb, emb, metric='euclidean')
    sep = np.square(sep)

    print "Dist to frontal scene 1: ", list(sep[0])
    print "Dist to pitch scene 1: ", list(sep[1])
    print "Dist to frontal scene 2: ", list(sep[2])
    print "Dist to pitch scene 2: ", list(sep[3])


    sep = pairwise_distances(emb[3], emb3, metric='euclidean')
    sep = np.square(sep)
    sep = sep[0]

    # best_index = np.argmin(sep)

    sep_sorted, sorted_indices = (list(t) for t in zip(*sorted(zip(sep, np.arange(0,len(sep))))))
    sep_sorted = np.array(sep_sorted)
    sorted_indices = np.array(sorted_indices)

    # mask = sep_sorted < 0.75
    # print sorted_indices[mask]

    print "best matches: ", sep_sorted[0:5]

    # display nearest X images
    display_image("matthias_test", sorted_indices[0:5], img_folder_name="matthias_test")


if __name__ == '__main__':

    calc_nearest_pose()
