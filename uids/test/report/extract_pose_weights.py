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

from sklearn.metrics.pairwise import *
from numpy import genfromtxt
import matplotlib.mlab as mlab
import random
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
import numpy.polynomial.polynomial as poly

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


class AngleGrid:

    grid = None
    grid_size = 0

    def __init__(self):
        pass

    def generate(self, embeddings, poses):

        d_p = d_y = 8

        if d_p == 8:
            min_i = -5
            max_i = 4
        elif d_p == 4:
            # -30 .. 30
            min_i = -8
            max_i = 7
        else:
            raise 0

        grid = []

        for i in range(min_i, max_i):
            p_lower = d_p / 2 * (1 + i * 2)
            p_upper = d_p / 2 * (1 + (i + 1) * 2)
            mask_pitch = (p_lower < poses[:, 2]) & (poses[:, 2] < p_upper)
            # print "--- {} ... {}: {} elems".format(p_lower, p_upper, np.count_nonzero(mask_pitch))

            const_pitch_embs = []

            for j in range(min_i, max_i):
                y_lower = d_y / 2 * (1 + j * 2)
                y_upper = d_y / 2 * (1 + (j + 1) * 2)
                mask_pitch = (p_lower < poses[:, 1]) & (poses[:, 1] < p_upper)
                mask_yaw = (y_lower < poses[:, 2]) & (poses[:, 2] < y_upper)
                mask = mask_pitch & mask_yaw
                nr_pictures = np.count_nonzero(mask)

                # print "P({}..{}), Y({}..{}): nr elems {}".format(p_lower, p_upper, y_lower, y_upper, nr_pictures)
                if nr_pictures == 0:
                    print "========= missing!"
                    const_pitch_embs.append([0])
                else:
                    const_pitch_embs.append(np.mean(embeddings[mask], axis=0))
            # add row
            # print "row items: ", len(const_pitch_embs)
            grid.append(const_pitch_embs)

        self.grid = np.array(grid)
        self.grid_size = self.grid.shape[0]

    def calc_index(self, pitch, yaw):

        d_p = d_y = 8.
        center = (self.grid_size-1)/2

        if pitch > 0:
            i = center + math.ceil((pitch - d_p / 2.) / d_p)
        elif pitch < 0:
            i = center - math.ceil((abs(pitch) - d_p / 2.) / d_p)
        else:
            i = center

        if yaw > 0:
            j = center + math.ceil((yaw - d_y / 2.) / d_y)
        elif yaw < 0:
            j = center - math.ceil((abs(yaw) - d_y / 2.) / d_y)
        else:
            j = center

        # pitch, yaw
        return int(i), int(j)

    def select(self, pitch, yaw):
        assert pitch < 36 and pitch > -36
        assert yaw < 36 and yaw > -36
        i, j = self.calc_index(pitch, yaw)
        return self.grid[i, j]

    def euclidean_dist(self, pitch1, yaw1, pitch2, yaw2):
        emb1 = self.select(pitch1, yaw1)
        emb2 = self.select(pitch2, yaw2)
        dist = pairwise_distances(emb1.reshape(1, -1), emb2.reshape(1, -1), metric='euclidean')[0][0]
        dist = np.square(dist)
        return dist

    def get_weight(self, pitch1, yaw1, pitch2, yaw2):
        pitch1 = np.clip(pitch1, -36, 36)
        yaw1 = np.clip(yaw1, -36, 36)
        pitch2 = np.clip(pitch2, -36, 36)
        yaw2 = np.clip(yaw2, -36, 36)

        dist = np.clip(self.euclidean_dist(pitch1, yaw1, pitch2, yaw2),0,1)
        weight = 1-dist
        return weight

    def test_indices(self):
        print "0", self.calc_index(-30, 20)
        print "1", self.calc_index(-28, 20)
        print "1", self.calc_index(-24, 20)
        print "2", self.calc_index(-20, 20)
        print "2", self.calc_index(-18, 20)
        print "3", self.grid.calc_index(-12, 20)
        print "3", self.calc_index(-8, 20)
        print "4", self.calc_index(-4, 20)
        print "4", self.calc_index(-1, 20)
        print "4", self.calc_index(2, 20)
        print "4", self.calc_index(4, 20)
        print "5", self.calc_index(8, 20)
        print "5", self.calc_index(12, 20)
        print "6", self.calc_index(18, 20)
        print "6", self.calc_index(20, 20)
        print "7", self.calc_index(24, 20)
        print "7", self.calc_index(27, 20)
        print "7", self.calc_index(28, 20)
        print "8", self.calc_index(28.5, 20)
        print "8", self.calc_index(30, 20)


def test2():

    # load embeddings
    emb = load_data('pose_matthias2.pkl')
    poses = load_data('pose_matthias2_poses.pkl')

    grid = AngleGrid()
    grid.generate(emb, poses)


    print grid.euclidean_dist(25,0, 0,0)
    print grid.euclidean_dist(25,10, 0,0)
    print grid.euclidean_dist(25,20, 0,0)
    print grid.euclidean_dist(25,30, 0,0)
    print grid.get_weight(25,30, 0,0)
    # print grid.euclidean_dist(30,0, 25,0)


if __name__ == '__main__':
    test2()
