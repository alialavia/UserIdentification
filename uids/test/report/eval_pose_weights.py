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
import time
import sys

from sklearn.metrics.pairwise import *
from numpy import genfromtxt
import matplotlib.mlab as mlab
import random
from uids.features.ConfidenceGen import WeightGenerator


# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory


def load_data(filename):
    filename = "{}/{}".format(modelDir, filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    else:
        print "File not found!"
    return None

def load_labels(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        my_data = genfromtxt(filename, delimiter=',')
        return my_data
    return None

# ------------------------------------------------

if __name__ == '__main__':

    gen1 = WeightGenerator(embedding_file='pose_matthias2.pkl', pose_file='pose_matthias2_poses.pkl')
    gen2 = WeightGenerator(embedding_file='christian_pose1.pkl', pose_file='christian_pose1_poses.pkl')
    gen3 = WeightGenerator(embedding_file='pose_elias.pkl', pose_file='pose_elias_poses.pkl')
    gen4 = WeightGenerator(embedding_file='pose_laia.pkl', pose_file='pose_laia_poses.pkl')
    gen5 = WeightGenerator(embedding_file='pose_matthias3.pkl', pose_file='pose_matthias3_poses.pkl')



    print "Top left: ", gen1.euclidean_dist([0,0], [30,-30])
    # print "Top left: ", gen2.euclidean_dist([0,0], [30,-30])
    # print "Top left: ", gen3.euclidean_dist([0,0], [30,-30])
    # print "Top left: ", gen4.euclidean_dist([0,0], [30,-30])
    print "Top left: ", gen5.euclidean_dist([0,0], [30,-30])


    ref_pose = np.array([20,20])

    plt.figure('Matthias')
    gen1.disp_heatmap(ref_pose)
    # plt.figure('Matthias - count map')
    # gen1.disp_count_heatmap()
    # plt.figure('Matthias - Variance')
    # gen1.disp_variance_heatmap()


    plt.figure('Matthias2')
    gen5.disp_heatmap(ref_pose)
    # plt.figure('Matthias2 - count map')
    # gen5.disp_count_heatmap()


    # plt.figure('Elias')
    # gen3.disp_heatmap(ref_pose)
    # plt.figure('Elias - Variance')
    # gen3.disp_variance_heatmap()

    plt.figure('Laia')
    gen4.disp_heatmap(ref_pose)
    # plt.figure('Laia - Variance')
    # gen4.disp_variance_heatmap()
    plt.figure('Laia - count map')
    gen4.disp_count_heatmap()

    plt.show()


