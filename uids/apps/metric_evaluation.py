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
modelDir = os.path.join(fileDir, '..', 'models', 'embedding_samples')	# path to the model directory
ressourceDir = os.path.join(fileDir, '..', 'ressource')	# path to the model directory


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


            # np.concatenate(())
            #
            #
            # a = [i.size for i in images]
            # print a
            #
            # widths, heights, dims = zip(*(i.shape for i in images))
            # total_width = sum(widths)
            # max_height = max(heights)
            #
            # new_im = Image.new('RGB', (total_width, max_height))
            #
            # x_offset = 0
            # for im in images:
            #     new_im.paste(im, (x_offset, 0))
            #     x_offset += im.size[0]

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


# ================================= #
#              Plotting

# Metrics:
# ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
# 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

def plot_dist_to_centroid(c1_samples, c2_samples, metric='cosine'):
    # calc centroid of fitting data
    mean = np.mean(c1_samples, axis=0)
    # std = np.std(c1_samples, axis=0)
    print "---metric: "+metric
    # inter-class centroid separation distribution
    intercl_dist = pairwise_distances(mean, c1_samples, metric=metric)
    max_intercl = intercl_dist.max(axis=1)
    min_intercl = intercl_dist.min(axis=1)

    # neighbour separation distribution
    out_dist = pairwise_distances(mean, c2_samples, metric=metric)
    max_out = out_dist.max(axis=1)
    min_out = out_dist.min(axis=1)

    print "--- min: {}, max: {}".format(min_intercl, max_intercl)

    # plot bins
    n, bins, patches = plt.hist(np.transpose(intercl_dist), 50, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(np.transpose(out_dist), 50, normed=1, facecolor='red', alpha=0.75)

    # add cluster indices
    k_list = [c1_samples, c2_samples]
    k_centers = [np.mean(c1_samples, axis=0), np.mean(c2_samples, axis=0)]

    # dunn = dunn_index(k_list, metric)
    # daviesbouldin = daviesbouldin_index(k_list, k_centers, metric)
    # plt.text(0.4, 5, "dunn: {}, db: {}".format(dunn, daviesbouldin), va='top', fontsize=9)

    # plot title, axis,
    plt.title('Metric evaluation: {}-distance'.format(metric))
    plt.ylabel('Number of samples')
    plt.xlabel('Distance from sample to class centroid')



# ================================= #
#        Evaluation Routines


def plot_inter_class_separation(ds1, ds2, metric='euclidean'):

    # Cosine distance is defined as 1.0 minus the cosine similarity.

    print "=====================================================\n"
    print "  INTER-CLASS SEPARATION IN {} DISTANCE\n".format(metric)
    print "=====================================================\n"

    if metric == 'cosine_similarity':
        sep = cosine_similarity(ds1, ds2)
    else:
        sep = pairwise_distances(ds1, ds2, metric=metric)

    sep = sep.flatten()

    max_out = np.amax(sep)
    min_out = np.amin(sep)
    mean = np.mean(sep)
    print "--- min: {}, max: {}, mean: {}".format(min_out, max_out, mean)

    fig = plt.figure()
    n, bins, patches = plt.hist(np.transpose(sep), 50, normed=1, facecolor='green', alpha=0.75)
    plt.title('Inter-Class separation: {}-distance'.format(metric))
    plt.ylabel('Number of samples')
    plt.xlabel('Sample separation')
    plt.show()


def plot_separation_dist(ds1, metric='euclidean', on_99_sub=False):

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

    if on_99_sub is False:
        plt.show()
        return

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


def print_distr_on_minvar_subspace(reference_ds, general_ds, d_eval_variance, explained_variance = 0.2, metric='euclidean'):

    print "=====================================================\n"
    print "  EVALUATE DISTRIBUTION MINIMAL VARIANCE ({}) SUBSPACE\n".format(explained_variance)
    print "=====================================================\n"

    # extract 99.9% subspace
    basis, mean = ExtractInverseSubspace(d_eval_variance, 1-explained_variance)
    print "--- reduced dimension to: {}".format(np.size(basis,1))

    # project data onto subspace
    d1_proj = ProjectOntoSubspace(reference_ds, mean, basis)
    d2_proj = ProjectOntoSubspace(general_ds, mean, basis)
    plot_dist_to_centroid(d1_proj, d2_proj, metric)

    k_list = []
    k_list.append(d1_proj)
    k_list.append(d2_proj)
    k_centers = []
    k_centers.append(np.mean(d1_proj, axis=0))
    k_centers.append(np.mean(d2_proj, axis=0))
    try:
        print "--- Dunn Index index: {}".format(dunn_index(k_list, metric))
        print "--- Davies Bouldin Index: {}".format(daviesbouldin_index(k_list, k_centers, metric))
    except MemoryError:
        pass
    print "=====================================================\n"
    plt.show()


def print_distr_on_subspace(reference_ds, general_ds, d_eval_variance, metric='euclidean'):

    print "=====================================================\n"
    print "        EVALUATE DISTRIBUTION ON 99.9% SUBSPACE\n"
    print "=====================================================\n"

    # extract 99.9% subspace
    basis, mean = ExtractSubspace(d_eval_variance, 0.999)
    print "--- reduced dimension to: {}".format(np.size(basis,1))

    # project data onto subspace
    d1_proj = ProjectOntoSubspace(reference_ds, mean, basis)
    d2_proj = ProjectOntoSubspace(general_ds, mean, basis)
    plot_dist_to_centroid(d1_proj, d2_proj, metric)

    k_list = []
    k_list.append(d1_proj)
    k_list.append(d2_proj)
    k_centers = []
    k_centers.append(np.mean(d1_proj, axis=0))
    k_centers.append(np.mean(d2_proj, axis=0))
    try:
        print "--- Dunn Index index: {}".format(dunn_index(k_list, metric))
        print "--- Davies Bouldin Index: {}".format(daviesbouldin_index(k_list, k_centers, metric))
    except MemoryError:
        pass
    print "=====================================================\n"
    plt.show()

def plot_different_distance_metrics(main_distr, general_distr):

    # start plotting
    fig = plt.figure()
    # min 128 samples: , 'mahalanobis'
    # metrics = ['euclidean', 'cosine', 'braycurtis', 'correlation', 'minkowski', 'seuclidean', 'sqeuclidean']
    metrics = ['euclidean', 'cosine']

    cols = 2.0
    rows = int(math.ceil(len(metrics) / cols))
    plot_format = str(rows)+str(int(cols))

    for i, m in enumerate(metrics):
        code = plot_format+str(i+1)
        try:
            ax = fig.add_subplot(int(code))
            plot_dist_to_centroid(main_distr, general_distr, m)
        except MemoryError:
            print "--- Out of memory: Could not calculate cluster metrics for {} metric".format(m)
        except Exception:
            print "--- Could not plot {} metric".format(m)
            continue

    plt.grid(True)
    plt.show()

# ================================= #
#              Main

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Image folder.", default="faces")
    parser.add_argument('--output', help="Statistics output folder.", default="stats")
    args = parser.parse_args()

    # load embeddings
    emb_1 = load_data('embeddings_matthias.pkl')
    emb_2 = load_data('embeddings_matthias_3.pkl')
    emb_3 = load_data('embeddings_matthias_clean.pkl')
    emb_4 = load_data('embeddings_matthias_logged.pkl')
    emb_5 = load_data('embeddings_elias.pkl')
    emb_6 = load_data('embeddings_laia.pkl')
    emb_lfw = load_data('embeddings_lfw.pkl')

    emb_combined = np.concatenate((emb_1, emb_2, emb_3, emb_4))

    if emb_1 is None or emb_2 is None:
        print "--- embeddings could not be loaded. Aborting..."
        return

    # ------------------- START EVALUATION
    # plot_separation_dist(emb_combined, 'cosine')
    # print_distr_on_minvar_subspace(emb_3, emb_lfw, emb_3, explained_variance = 1.0, metric='cosine')
    # print_distr_on_minvar_subspace(emb_3, emb_lfw, emb_3, explained_variance = 0.05, metric='cosine')



    if False:
        # extract 20% subspace
        basis, mean = ExtractInverseSubspace(emb_3, 1-0.2)
        print "--- reduced dimension to: {}".format(np.size(basis, 1))

        # project data onto subspace
        emb_3 = ProjectOntoSubspace(emb_3, mean, basis)
        plot_separation_dist(emb_2, 'cosine')
        emb_2 = ProjectOntoSubspace(emb_2, mean, basis)
        plot_separation_dist(emb_2, 'cosine')
    if True:
        # extract 20% subspace
        basis, mean = ExtractInverseSubspace(emb_3, 1-0.1)
        print "--- reduced dimension to: {}".format(np.size(basis, 1))


        # project data onto subspace
        emb_3 = ProjectOntoSubspace(emb_3, mean, basis)
        emb_5 = ProjectOntoSubspace(emb_5, mean, basis)

        plot_separation_dist(emb_3, 'cosine')
        plot_ds_separations(emb_3, emb_5, 'cosine')

    return

    # 1. DISTANCE TO CENTROID

    # 1.1 Plot different distance metrics
    plot_different_distance_metrics(emb_1, emb_2)

    # 1.2 Plot distance to centroid on 99.9% general subspace
    print_distr_on_subspace(emb_1, emb_2, emb_lfw, 'cosine')

    # 2. INTRA-/INTER-CLASS DISTANCE DISTRIBUTION
    plot_intra_class_separation_on_class_subspace(emb_1, 'cosine')
    plot_inter_class_separation(emb_1, emb_lfw, 'cosine')


def run_evaluation_2():
    emb1 = load_data("embeddings_elias.pkl")
    emb2 = load_data("embeddings_matthias_big.pkl")
    emb3 = load_data("embeddings_laia.pkl")
    emb_lfw = load_data("embeddings_lfw.pkl")

    # filter blurred images
    l = load_labels('blur_labels_matthias_big.csv')
    l = l[:,1]
    blurred = emb2[l==1]
    clear = emb2[l==0]

    print len(blurred)
    print len(clear)

    # plot separations
    plot_inter_class_separation(clear, clear, 'cosine')
    # plot_inter_class_separation(clear, blurred, 'cosine')



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


def calc_embedding_corr():

    metric = 'euclidean'

    # select range
    # emb_yaw = emb_yaw[0:-2,:]
    # emb_pitch = emb_pitch[1:,:]

    sep = pairwise_distances(emb[0], emb[1:], metric='euclidean')
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
    # plot_pitch_yaw_comparison()
    # plot_shading_variance()
    # plot_background_influence()
    # emb = load_data("color_embeddings.pkl")
    # sep = pairwise_distances(emb[0,:], emb[1:,:], metric='cosine')
    # sep = pairwise_distances(emb[0,:], emb[1:,:], metric='cosine')
    # print sep
    # sep = pairwise_distances(emb[1,:], emb[4,:], metric='cosine')
    # print sep
    # sep = pairwise_distances(emb[2,:], emb[5,:], metric='cosine')

    calc_nearest_pose()

    # # load embeddings
    # emb_1 = load_data('embeddings_matthias.pkl')
    # emb_2 = load_data('embeddings_matthias_3.pkl')
    # emb_3 = load_data('embeddings_matthias_clean.pkl')
    # emb_4 = load_data('embeddings_matthias_logged.pkl')
    # emb_5 = load_data('embeddings_elias.pkl')
    # emb_6 = load_data('embeddings_laia.pkl')
    # emb_lfw = load_data('embeddings_lfw.pkl')
    # plot_inter_class_separation(emb_1, emb_lfw)
