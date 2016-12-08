#!/usr/bin/env python2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.extmath import fast_dot

# ================================= #
#              Data Analysis

def ExtractSubspace(data, explained_variance):
    """"""
    pca = PCA(n_components=np.size(data, 1))
    pca.fit(data)
    var_listing = pca.explained_variance_ratio_

    s = 0
    index = 0
    for i, v in enumerate(var_listing):
        if s + v > explained_variance:
            break
        s = s + v
        index = i

    # extract basis
    basis = pca.components_.T[:, 0:(index+1)]
    return basis, pca.mean_


def ProjectOntoSubspace(data, mean, basis):
    """Applie simension reduction"""
    reduced = data - mean
    reduced = fast_dot(reduced, basis)

    reduced = reduced + fast_dot(mean, basis)

    return reduced


def CalcComponentVariance(v):
    """Calculate Eigenbasis and the Eigenvector contributions"""
    pca = PCA(n_components=np.size(v,1))
    pca.fit(v)
    return pca.explained_variance_ratio_


def ExtractInverseSubspace(data, explained_variance):
    """"""
    pca = PCA(n_components=np.size(data, 1))
    pca.fit(data)
    var_listing = pca.explained_variance_ratio_

    s = 0
    index = 0
    for i, v in enumerate(var_listing):
        if s + v > explained_variance:
            break
        s = s + v
        index = i

    # extract basis inverse
    basis = pca.components_.T[:, (index+1):np.size(data, 1)]
    return basis, pca.mean_

