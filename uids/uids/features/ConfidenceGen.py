#!/usr/bin/env python2
import os
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import *
import math
import time
from sklearn.metrics.pairwise import *
from uids.utils.Logger import Logger as log
import matplotlib.pyplot as plt


# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'confidence_weights') # path to the model directory


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


# ------------------------------------------------

class WeightGenerator:

    grid = None
    variance_map = None
    count_map = None

    grid_size = 0
    d_p = None
    d_y = None

    def __init__(self, embedding_file="pose_matthias2.pkl", pose_file="pose_matthias2_poses.pkl"):

        log.info('db', "Initializing weight generator...")
        # initialize grid
        embeddings = load_data(embedding_file)
        poses = load_data(pose_file)
        self.generate(embeddings, poses)

    def generate(self, embeddings, poses):

        self.d_p = self.d_y = 8

        d_p = self.d_p
        d_y = self.d_y

        if d_p == 8:
            # -36 .. 36
            min_i = -5
            max_i = 4
        elif d_p == 4:
            # -30 .. 30
            min_i = -8
            max_i = 7
        else:
            raise 0

        grid = []
        variance = []
        count = []

        for i in range(min_i, max_i):
            p_lower = d_p / 2 * (1 + i * 2)
            p_upper = d_p / 2 * (1 + (i + 1) * 2)
            # print "--- {} ... {}: {} elems".format(p_lower, p_upper, np.count_nonzero(mask_pitch))

            const_pitch_embs = []
            const_pitch_emb_variance = []
            const_pitch_count = []

            for j in range(min_i, max_i):
                y_lower = d_y / 2 * (1 + j * 2)
                y_upper = d_y / 2 * (1 + (j + 1) * 2)
                mask_pitch = (p_lower < poses[:, 1]) & (poses[:, 1] < p_upper)
                mask_yaw = (y_lower < poses[:, 2]) & (poses[:, 2] < y_upper)
                mask = mask_pitch & mask_yaw
                nr_pictures = np.count_nonzero(mask)
                const_pitch_count.append(nr_pictures)

                if nr_pictures == 0:
                    print "P({}..{}), Y({}..{}): nr elems {}".format(p_lower, p_upper, y_lower, y_upper, nr_pictures)
                    # raise "========= missing!"
                    const_pitch_embs.append(np.array([0]*128))
                    const_pitch_emb_variance.append(0)
                else:
                    const_pitch_embs.append(np.mean(embeddings[mask], axis=0))
                    dist = pairwise_distances(embeddings[mask], embeddings[mask], metric='euclidean')
                    dist = np.square(dist)
                    const_pitch_emb_variance.append(np.var(dist))

            # add row
            # print "row items: ", len(const_pitch_embs)
            grid.append(np.array(const_pitch_embs))
            variance.append(np.array(const_pitch_emb_variance))
            count.append(np.array(const_pitch_count))

        self.count_map = np.array(count)
        self.variance_map = np.array(variance)
        self.grid = np.array(grid)
        self.grid_size = self.grid.shape[0]

    def calc_index(self, pitch, yaw):
        d_p = d_y = self.d_p
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

    def disp_count_heatmap(self):
        var = np.fliplr(np.flipud(self.count_map))
        plt.imshow(var, cmap='GnBu_r', interpolation='nearest')
        cbar = plt.colorbar()
        cl = plt.getp(cbar.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize=16)

    def disp_variance_heatmap(self):
        var = np.fliplr(np.flipud(self.variance_map))
        plt.imshow(var, cmap='GnBu_r', interpolation='nearest')
        cbar = plt.colorbar()
        cl = plt.getp(cbar.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize=16)

    def disp_heatmap(self, ref_pitch_yaw):
        sep = self.get_dist_matrix(ref_pitch_yaw)
        sep = np.fliplr(np.flipud(sep))
        plt.imshow(sep, cmap='GnBu_r', interpolation='nearest')
        cbar = plt.colorbar()
        cl = plt.getp(cbar.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize=16)

    def select(self, pitch_yaw):
        pitch_yaw = np.clip(pitch_yaw, -36, 36)
        pitch = pitch_yaw[0]
        yaw = pitch_yaw[1]

        # assert pitch < 36 and pitch > -36
        # assert yaw < 36 and yaw > -36
        i, j = self.calc_index(pitch, yaw)
        return self.grid[i, j]

    def euclidean_dist_squared(self, pitch_yaw1, pitch_yaw2):
        emb1 = self.select(pitch_yaw1)
        emb2 = self.select(pitch_yaw2)
        dist = pairwise_distances(emb1.reshape(1, -1), emb2.reshape(1, -1), metric='euclidean')[0][0]
        dist = np.square(dist)
        return dist

    def euclidean_dist(self, pitch_yaw1, pitch_yaw2):
        emb1 = self.select(pitch_yaw1)
        emb2 = self.select(pitch_yaw2)
        dist = pairwise_distances(emb1.reshape(1, -1), emb2.reshape(1, -1), metric='euclidean')[0][0]
        return dist

    # get dist weight
    def get_dist_weight_clipped(self, pitch_yaw1, pitch_yaw2):
        emb1 = self.select(pitch_yaw1)
        emb2 = self.select(pitch_yaw2)
        dist = pairwise_distances(emb1.reshape(1, -1), emb2.reshape(1, -1), metric='euclidean')[0][0]
        dist = np.clip(dist, 0, 1) + 1
        return dist

    def get_dist_matrix(self, pitch_yaw1):
        emb_ref = self.select(pitch_yaw1)
        g_tmp = self.grid.reshape(81,128)
        dist = pairwise_distances(emb_ref, g_tmp, metric='euclidean')
        dist = np.square(dist)
        return dist.reshape(9, 9)

    def get_weight(self, pitch_yaw1, pitch_yaw2):

        pitch_yaw1 = np.clip(pitch_yaw1, -36, 36)
        pitch_yaw2 = np.clip(pitch_yaw2, -36, 36)
        dist = np.clip(self.euclidean_dist_squared(pitch_yaw1, pitch_yaw2), 0, 1)
        if dist < 0.1:
            pass
            # print pitch_yaw1, pitch_yaw2

        if dist > 0.2:
            pass
            # print pitch_yaw1, pitch_yaw2

        # calculate weight
        # weight = 100*(1-dist**2)
        weight = 1-dist

        # clip to range
        weight = np.clip(weight, 0.01,1)

        # print dist
        return weight

    def get_triplet_weight(self, pitch_yaw_ref, pitch_yaw1, pitch_yaw2):
        # w_touple = w1 * w2
        return self.get_weight(pitch_yaw_ref, pitch_yaw1)*self.get_weight(pitch_yaw_ref, pitch_yaw2)
        # return np.min(self.get_weight(pitch_yaw_ref, pitch_yaw1), self.get_weight(pitch_yaw_ref, pitch_yaw2))

    def test_indices(self):
        print "0", self.calc_index(-30, 20)
        print "1", self.calc_index(-28, 20)
        print "1", self.calc_index(-24, 20)
        print "2", self.calc_index(-20, 20)
        print "2", self.calc_index(-18, 20)
        print "3", self.calc_index(-12, 20)
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

    def best_subset(self, test_pose, ref_poses, nr_samples=20):
        # take 30 nearest samples from ref

        # calc dist test > ref
        emb_test = self.select(test_pose)
        # select ref embeddings
        emb_ref = [self.select(p) for p in ref_poses]

        # calc dist
        dist = pairwise_distances(emb_test.reshape(1, -1), emb_ref, metric='euclidean')[0]
        dist = np.square(dist)

        sorted_indices = dist.argsort()
        nr_ref_elems = nr_samples if len(ref_poses) >= nr_samples else len(ref_poses)

        # print "Distances: ", dist[sorted_indices[0:nr_ref_elems]]
        return sorted_indices[0:nr_ref_elems], dist[sorted_indices]

