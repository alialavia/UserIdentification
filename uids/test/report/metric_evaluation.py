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
    return None


def load_labels(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        my_data = genfromtxt(filename, delimiter=',')
        return my_data
    return None


# ================================= #
#              Plotting

# ================================= #
#        Evaluation Routines


def plot_inter_class_separation(ds1, ds2, metric='euclidean'):

    # Cosine distance is defined as 1.0 minus the cosine similarity.

    print "=====================================================\n"
    print "  INTER-CLASS SEPARATION IN {} DISTANCE\n".format(metric)
    print "=====================================================\n"

    if metric == 'cosine_similarity':
        sep_intra = cosine_similarity(ds1, ds1)
        sep_inter = cosine_similarity(ds1, ds2)
    else:
        sep_intra = pairwise_distances(ds1, ds1, metric=metric)
        sep_inter = pairwise_distances(ds1, ds2, metric=metric)

    sep_intra = sep_intra.flatten()
    sep_inter = sep_inter.flatten()


    max_out = np.amax(sep_intra)
    min_out = np.amin(sep_intra)
    mean = np.mean(sep_intra)


    print "---sep_intra: min: {}, max: {}, mean: {}".format(min_out, max_out, mean)

    fig = plt.figure()
    n, bins, patches = plt.hist(np.transpose(sep_inter), 50, normed=1, facecolor='#069af3', alpha=0.75)
    n, bins, patches = plt.hist(np.transpose(sep_intra), 50, normed=1, facecolor='#00B050', alpha=0.75)


    # plt.hist(np.transpose(sep_intra), bins=20)



    # plt.title('Inter-Class separation: {}-distance'.format(metric))
    plt.ylabel('Number of samples')
    plt.xlabel('Sample separation')
    plt.show()

def density_estimation():
    from scipy.stats.distributions import norm

    # The grid we'll use for plotting
    x_grid = np.linspace(-4.5, 3.5, 1000)

    # Draw points from a bimodal distribution in 1D
    np.random.seed(0)
    x = np.concatenate([norm(-1, 1.).rvs(400),
                        norm(1, 0.3).rvs(100)])
    pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) +
                0.2 * norm(1, 0.3).pdf(x_grid))

    # Plot the three kernel density estimates
    fig, ax = plt.subplots(1, 4, sharey=True,
                           figsize=(13, 3))
    fig.subplots_adjust(wspace=0)

    pdf = kde_funcs[i](x, x_grid, bandwidth=0.2)
    ax[i].plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    ax[i].fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    ax[i].set_title(kde_funcnames[i])
    ax[i].set_xlim(-4.5, 3.5)


def density_est_kde(ds1, ds2, metric='euclidean'):

    if metric == 'cosine_similarity':
        sep_intra = cosine_similarity(ds1, ds1)
        sep_inter = cosine_similarity(ds1, ds2)
    else:
        sep_intra = pairwise_distances(ds1, ds1, metric=metric)
        sep_inter = pairwise_distances(ds1, ds2, metric=metric)

    sep_intra = sep_intra.flatten()
    sep_inter = sep_inter.flatten()

    # --- intra

    xfit = np.linspace(0, 2, len(sep_intra))
    X = sep_intra[:, np.newaxis]
    Xfit = xfit[:, np.newaxis]
    kde = KernelDensity(bandwidth=0.05)
    kde.fit(X)
    log_dens = kde.score_samples(Xfit)
    density = np.exp(log_dens)
    # density *= 1/np.sum(density)
    plt.plot(xfit, density, '#069af3', lw=2)

    print "--- fitted intra cluster separation"

    # --- inter

    xfit = np.linspace(0, 2, len(sep_inter))
    X = sep_inter[:, np.newaxis]
    Xfit = xfit[:, np.newaxis]
    kde = KernelDensity(bandwidth=0.05)
    kde.fit(X)
    log_dens = kde.score_samples(Xfit)
    density = np.exp(log_dens)
    # density *= 1/np.sum(density)
    plt.plot(xfit, density, '#00B050', lw=2)

    plt.show()






def density_est_histo(metric, ref_data, *comparison_data):

    bin_nr = 50
    plot_values = True
    plot_gaussian = True

    if metric == 'cosine_similarity':
        sep_intra = cosine_similarity(ref_data, ref_data)
    else:
        sep_intra = pairwise_distances(ref_data, ref_data, metric=metric)

    sep_intra = sep_intra[sep_intra > 0]
    sep_intra = sep_intra.flatten()

    max_out = np.amax(sep_intra)
    min_out = np.amin(sep_intra)
    mean = np.mean(sep_intra)
    print "--- sep_intra: min: {}, max: {}, mean: {}, variance: {}, std: {}".format(min_out, max_out, mean, np.var(sep_intra), np.std(sep_intra))

    plt.figure()

    # 069af3
    n, bins, patches = plt.hist(np.transpose(sep_intra), bin_nr, normed=1, facecolor='white', edgecolor='none', alpha=0.75)
    print "--- bin width: {}".format(bins[1] - bins[0])

    plt.text(.5, 2.7, r'$\mu_1={:.2},\ \sigma_1={:.2}$'.format(mean, np.std(sep_intra)), fontsize=15)

    if plot_values:
        bin_width = bins[1]-bins[0]
        x = bins[:-1]+bin_width/2
        plt.plot(x, n, lw=3, color='#0771b1')
        print list(x)
        print list(n)

    if plot_gaussian:
        # best fit of data
        (mu, sigma) = norm.fit(sep_intra)
        y = mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2, color='#0771b1')

    for k, dataset in enumerate(comparison_data):
        if metric == 'cosine_similarity':
            sep_inter = cosine_similarity(ref_data, dataset)
        else:
            sep_inter = pairwise_distances(ref_data, dataset, metric=metric)

        sep_inter = sep_inter.flatten()

        max_out = np.amax(sep_inter)
        min_out = np.amin(sep_inter)
        mean = np.mean(sep_inter)
        print "--- sep_inter ds{}: min: {}, max: {}, mean: {}, variance: {}, std: {}".format(k, min_out, max_out, mean, np.var(sep_inter), np.std(sep_inter))

        color = '#00b050'
        line_clr = '#00b050'
        if k==0:
            plt.text(1.05, 3.4, r'$\mu_2={:.2},\ \sigma_2={:.2}$'.format(np.mean(sep_inter), np.std(sep_inter)), fontsize=15)
        elif k==1:
            color = '#92d050'
            line_clr = '#92d050'
            plt.text(1.34, 2.9, r'$\mu_3={:.2},\ \sigma_3={:.2}$'.format(np.mean(sep_inter), np.std(sep_inter)), fontsize=15)
        elif k==2:
            color = '#92D050'
            line_clr = '#628839'

        color = 'white'

        n, bins, patches = plt.hist(np.transpose(sep_inter), bin_nr, normed=1, facecolor=color, edgecolor='none', alpha=0.9)
        print "--- bin width: {}".format(bins[1] - bins[0])

        if plot_values:
            bin_width = bins[1] - bins[0]
            x = bins[:-1] + bin_width / 2
            plt.plot(x, n, lw=3, color=line_clr)

            print list(x)
            print list(n)

        if plot_gaussian:
            # best fit of data
            (mu, sigma) = norm.fit(sep_inter)
            y = mlab.normpdf(bins, mu, sigma)
            l = plt.plot(bins, y, '--', color=line_clr, linewidth=2)



    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Probability Density Function (PDF)')
    plt.title('Intra- and Inter-Class Distance Distribution')
    plt.ylim([0,3.6])
    plt.grid()

    plt.rcParams.update({'font.size': 14})


    # plt.axis([40, 160, 0, 0.03])


def build_sum_timeseries(ts, bin_width):
    sum = []
    for t, dist in enumerate(ts):
        if t == 0:
            sum.append(ts[0]*bin_width)
        else:
            sum.append(sum[t-1]+ts[t]*bin_width)
    return sum



def density_overlapp(metric, ref_data, *comparison_data):

    bin_nr = 50
    plot_values = True
    plot_gaussian = True

    sep_intra = pairwise_distances(ref_data, ref_data, metric=metric)
    sep_intra = sep_intra[sep_intra > 0]
    sep_intra = sep_intra.flatten()

    max_out = np.amax(sep_intra)
    min_out = np.amin(sep_intra)
    mean = np.mean(sep_intra)
    print "--- sep_intra: min: {}, max: {}, mean: {}, variance: {}, std: {}".format(min_out, max_out, mean, np.var(sep_intra), np.std(sep_intra))

    # first
    n, bins, patches = plt.hist(np.transpose(sep_intra), bin_nr, normed=1, facecolor='white', edgecolor='none', alpha=0.75)
    print "--- bin width: {}".format(bins[1] - bins[0])

    bin_width = bins[1] - bins[0]
    x = bins[:-1] + bin_width / 2
    # plt.plot(x, n, lw=3, color='#0771b1')

    print list(x)
    print build_sum_timeseries(n, bin_width)


    plt.plot(x,build_sum_timeseries(n, bin_width))

    for k, dataset in enumerate(comparison_data):
        sep_inter = pairwise_distances(ref_data, dataset, metric=metric)
        sep_inter = sep_inter.flatten()
        n, bins, patches = plt.hist(np.transpose(sep_inter), bin_nr, normed=1, facecolor='white', edgecolor='none', alpha=0.9)

        bin_width = bins[1] - bins[0]
        x = bins[:-1] + bin_width / 2
        # plt.plot(x, n, lw=3, color='#0771b1')
        plt.plot(x, build_sum_timeseries(n, bin_width))

        print list(x)
        print build_sum_timeseries(n, bin_width)


    plt.axhline(y=0.95)

    # 20 % overlapp
    plt.axhline(y=0.2)
    plt.axhline(y=0.976)
    plt.axvline(x=1.05)

    # stated boundary
    plt.axvline(x=0.99)

    plt.ylim([0,1])
    plt.ylabel('Cumulative Distribution Function (CDF)')

def kde_example():

    # ----------------------------------------------------------------------
    # Plot a 1D density example
    N = 100
    np.random.seed(1)
    X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
                        np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]

    X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

    true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
                 + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

    fig, ax = plt.subplots()
    ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
            label='input distribution')

    for kernel in ['gaussian', 'tophat', 'epanechnikov']:
        kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
        log_dens = kde.score_samples(X_plot)
        ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                label="kernel = '{0}'".format(kernel))

    ax.text(6, 0.38, "N={0} points".format(N))

    ax.legend(loc='upper left')
    ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

    ax.set_xlim(-4, 9)
    ax.set_ylim(-0.02, 0.4)
    plt.show()

if __name__ == '__main__':

    # load embeddings
    emb_1 = load_data('embeddings_matthias.pkl')
    emb_2 = load_data('embeddings_matthias_big.pkl')
    emb_3 = load_data('embeddings_matthias_clean.pkl')
    emb_4 = load_data('embeddings_christian.pkl')
    emb_5 = load_data('embeddings_christian_clean.pkl')
    emb_6 = load_data('embeddings_laia.pkl')
    emb_lfw = load_data('embeddings_lfw.pkl')

    # random.shuffle(emb_lfw)

    # density_est_kde(emb_2, emb_lfw[0:5000,:])

    # plot histogram density estimation
    density_est_histo('euclidean', emb_2, emb_5, emb_lfw)

    # plot cumulative density function
    # density_overlapp('euclidean', emb_2, emb_5, emb_lfw)

    plt.show()

    # kde_example()
    # plot_inter_class_separation(emb_2, emb_lfw)
    #
    # X_plot = np.linspace(0, 1.7,num=len(emb_2))
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(emb_2)
    # score = kde.score_samples(X_plot)
    # plt.plot(X_plot, score)