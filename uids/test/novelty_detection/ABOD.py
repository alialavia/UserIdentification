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


def load_labels(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        my_data = genfromtxt(filename, delimiter=',')
        return my_data
    return None


# ================================= #
#        Test Functions

def test_ABOD():

    clf = ABOD()

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_matthias_big.pkl")
    emb4 = load_embeddings("embeddings_laia.pkl")
    emb5 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    clf.fit(emb2)

    # class_sample = emb3[100,:]
    # outlier_sample = emb1[30,:]

    # print class_sample

    start = time.time()
    abod_class = clf.predict_approx(emb3)
    print "time: ".format(time.time()-start)

    return
    abod_outliers = clf.predict(emb5)
    step = 0.0001
    start = 0.005
    stop = 0.6
    il = []
    ul = []
    x = np.arange(start, stop, step)
    for thresh in x:
        il.append(float(len(abod_class[abod_class<thresh]))/len(abod_class)*100.0)
        ul.append(float(len(abod_outliers[abod_outliers>thresh]))/len(abod_outliers)*100.0)

    plt.plot(x,il,color='green', label="Inliers")
    plt.plot(x,ul,color='red', label="Outliers")
    plt.title("Classification Error")
    plt.xlabel("Threshold")
    plt.ylabel("Error [%]")
    plt.legend()
    plt.show()

    #
    thresh = 0.2
    print "error il: {}/{} : {}%".format(len(abod_class[abod_class<0.2]), len(abod_class),float(len(abod_class[abod_class<0.2]))/len(abod_class)*100.0)
    print "error ul: {}/{} : {}%".format(len(abod_outliers[abod_outliers>0.2]), len(abod_outliers),float(len(abod_outliers[abod_outliers>0.2]))/len(abod_outliers)*100.0)


# ================================= #
#              Main

if __name__ == '__main__':
    test_ABOD()
