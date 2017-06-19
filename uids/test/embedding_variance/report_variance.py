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



def load_labels(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        my_data = genfromtxt(filename, delimiter=',')
        return my_data
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

def calc_nearest_pose():

    emb = load_data("compare_nearest_pose.pkl")
    emb2 = load_data("christian_clean.pkl")
    emb3 = load_data("matthias_test.pkl")


    sep = pairwise_distances(emb, emb, metric='euclidean')
    sep = np.square(sep)


def plot_background_influence():
    emb = load_data("mat_bg.pkl")
    neutral = emb[0,:]
    rest = emb[1:,:]
    metric = 'cosine'

    # best order
    # switched = np.array([emb[2,:], emb[0,:], emb[3,:], emb[1,:], emb[4,:]])

    sep = pairwise_distances(emb, emb, metric=metric)

    # plt.figure()
    # plt.imshow(sep, cmap='GnBu', interpolation='nearest')
    # plt.colorbar()
    plt.figure()
    plt.imshow(sep, cmap='Blues_r', interpolation='nearest')
    cbar = plt.colorbar()

    cl = plt.getp(cbar.ax, 'ymajorticklabels')
    plt.setp(cl, fontsize=16)

    if metric == 'cosine':
        # cbar.set_ticks([0, 0.05, 0.1, 0.15, 0.2])
        pass
    else:
        cbar.set_ticks([0,0.2, 0.4, 0.6])

    plt.show()
    print sep


def plot_shading_variance():

    if True:
        print "======================================"
        print "       Evaluating general variance approximation...\n"
        emb = load_data("variance_eval/light_influence_general.pkl")
        sep = pairwise_distances(emb[2:3], emb, metric='euclidean')
        sep = np.square(sep)

        plt.title("Deviation to neutral lighting")
        plt.plot(np.arange(0,len(emb)), sep[0])
        plt.ylabel("Deviation [squared L2 distance]")
        plt.show()

        print "Deviation to neutral", list(sep[0])

    if True:
        print "======================================"
        print "       Evaluating extreme cases...\n"

        emb = load_data("variance_eval/shading.pkl")

        # best order
        # switched = np.array([emb[0,:], emb[2,:], emb[3,:], emb[1,:], emb[4,:]])
        sep = pairwise_distances(emb, emb, metric='euclidean')
        sep = np.square(sep)

        print "Deviation to neutral: ", list(sep[0])
        # remove autocorrelation
        # sep[sep==0] = sep[sep !=0 ].min()
        plt.figure()
        plt.imshow(sep, cmap='Blues_r', interpolation='nearest')
        cbar = plt.colorbar()
        cl = plt.getp(cbar.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize=16)
        cbar.set_ticks([0,0.2, 0.4, 0.6])
        plt.show()


def plot_facial_expression_dist():
    emb = load_data("variance_eval/facial_expressions.pkl")
    neutral = emb[0,:]
    rest = emb[1:,:]

    # best order
    switched = np.array([emb[0,:], emb[2,:], emb[3,:], emb[1,:], emb[4,:]])

    sep = pairwise_distances(emb, emb, metric='euclidean')
    sep = np.square(sep)

    print "Deviation to neutral: ", sep[0]

    sep2 = pairwise_distances(switched, switched, metric='euclidean')
    sep2 = np.square(sep2)

    sep2 = np.flipud(sep2)

    # remove autocorrelation
    sep2[sep2==0] = sep2[sep2 !=0 ].min()

    # plt.figure()
    # plt.imshow(sep, cmap='GnBu', interpolation='nearest')
    # plt.colorbar()
    plt.figure("Facial Expression Influence")
    plt.imshow(sep2, cmap='GnBu_r', interpolation='nearest')
    # plt.imshow(sep2, cmap='Blues_r', interpolation='nearest')
    cbar = plt.colorbar()

    cl = plt.getp(cbar.ax, 'ymajorticklabels')
    plt.setp(cl, fontsize=16)

    cbar.set_ticks([0,0.2, 0.3, 0.4])

    plt.show()


def run_blur_evaluation():

    # 0: no blur, 1: 1.5, 2: 3, 4: 4.5
    emb1 = load_data("embedding_frontal_increasing_blur.pkl")
    emb2 = load_data("embeddings_matthias_big.pkl")
    # filter blurred images
    # l = load_labels('blur_labels_matthias_big.csv')
    # l = l[:,1]
    # blurred = emb2[l==1]
    # clear = emb2[l==0]

    # 1: Draw Separation of a Frontal Face to Blurred Versions
    if True:
        blurred = load_data("variance_eval/gaussian_blur.pkl")
        sep = pairwise_distances(blurred[0:1,:], blurred[1:,:], metric='euclidean')
        sep = np.square(sep)
        blur_stages = [0.5, 1, 1.5, 2]
        plt.title('Separation of blurred (Gaussian) to unblurred Image')
        plt.ylabel('Separation [Squared L2 dist]')
        plt.xlabel('Blur Radius [px]')
        print "Deviation through Gaussian blur: ", sep[0]
        print "Blur stages: ", blur_stages
        plt.plot(blur_stages, sep[0])
        plt.show()

    # 1: Draw Separation of a Frontal Face to Blurred Versions
    if True:
        blurred = load_data("variance_eval/motion_blur.pkl")
        sep = pairwise_distances(blurred[0:1,:], blurred[1:,:], metric='euclidean')
        sep = np.square(sep)
        blur_stages = [2, 4, 6, 8]
        plt.title('Separation of blurred (Motion) to unblurred Image')
        plt.ylabel('Separation [Squared L2 dist]')
        plt.xlabel('Motion Blur Magnitude [px]')
        print "Deviation through Gaussian blur: ", sep[0]
        print "Blur stages: ", blur_stages
        plt.plot(blur_stages, sep[0])
        plt.show()


    # 2: Draw Separation of Blurred Frontal View to General Dataset
    blur_stages = [0, 1.5, 3, 4.5]
    if False:
        for i, blur in enumerate(blur_stages):
            plot_inter_class_separation(emb1[i,:], emb2, 'cosine')

    # 3: Calculate Distribution Statistics for Separation of Frontal View vs General Dataset
    if False:
        min = []
        max = []
        mean = []
        median = []
        for i, blur in enumerate(blur_stages):
            sep = pairwise_distances(emb1[i,:], emb2, metric='cosine')
            min.append(np.amin(sep))
            max.append(np.amax(sep))
            mean.append(np.mean(sep))
            median.append(np.median(sep))

        plt.title('Separation of Blurred Frontal View to Random Class Images')
        plt.ylabel('Separation [cosine distance]')
        plt.xlabel('Gaussian Blur Radius [px]')
        plt.plot(blur_stages, min, label="Minimum")
        plt.plot(blur_stages, max, label="Maximum")
        plt.plot(blur_stages, mean, label="Mean")
        plt.plot(blur_stages, median, label="Median")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # plot_facial_expression_dist()
    plot_shading_variance()
    # run_blur_evaluation()
