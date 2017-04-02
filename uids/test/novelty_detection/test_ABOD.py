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
from sklearn.svm import SVC
from uids.online_learning.BinaryThreshold import BinaryThreshold

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

    clf.fit(emb2[0:100])

    # class_sample = emb3[100,:]
    # outlier_sample = emb1[30,:]

    abod_class = clf.predict(emb3[0:100])

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


def test1():
    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_matthias_big.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb4 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    clf = SVC(kernel='linear', probability=True)
    clf2 = ABOD()

    # train user and unknown class
    label_class = np.repeat(1, np.shape(emb1[0:100])[0])
    label_unknown = np.repeat(0, np.shape(emb_lfw)[0])
    training_embeddings = np.concatenate((emb1[0:100], emb_lfw))
    training_labels = np.concatenate((label_class, label_unknown))
    # train svm
    clf.fit(training_embeddings, training_labels)
    # train abod
    clf2.fit(emb1[0:100])


    # test on class
    prediction = clf.predict(emb2[0:100])
    errors = len(emb2[0:100])-np.sum(prediction)
    print "Error rate: {}%".format(float(errors)/len(emb2[0:100])*100.0)

    # test on similar class
    prediction = clf.predict(emb4)
    errors = np.sum(prediction)
    print "Error rate: {}%".format(float(errors)/len(emb4)*100.0)


def cascaded_classifiers():
    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_matthias_big.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb4 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    clf = SVC(kernel='linear', probability=True, C=1)
    clf2 = ABOD()

    # random.shuffle(emb1)

    train = emb1[0:50]
    test = emb2
    ul = emb4

    # train user and unknown class
    label_class = np.repeat(1, np.shape(train)[0])
    label_unknown = np.repeat(0, np.shape(emb_lfw)[0])
    training_embeddings = np.concatenate((train, emb_lfw))
    training_labels = np.concatenate((label_class, label_unknown))
    clf.fit(training_embeddings, training_labels)
    clf2.fit(train)

    # --------------------- test on class
    prediction = clf.predict(test)
    errors = len(test)-np.sum(prediction)
    print "SVM Error rate: {}%".format(float(errors)/len(test)*100.0)
    temp = test
    # filter samples classified as 'unknown'
    filtered = temp[prediction == 0]
    # eval on abod
    abod_values = clf2.predict(filtered)
    errors = abod_values[abod_values < 0]
    print "Total error (inliers classified as outliers): {}%".format(float((len(errors))/float(len(test))))
    print "{}/{} additional inliers have been detected".format(len(abod_values[abod_values > 0]), len(filtered))

    # --------------------- test on outlier
    print "-------------testing on outliers----------------"
    prediction = clf.predict(ul)
    errors = np.sum(prediction)
    print "SVM Error rate: {}%".format(float(errors) / len(ul) * 100.0)
    temp = ul
    # filter samples classified as 'inliers'
    filtered = temp[prediction == 1]
    # eval on abod
    abod_values = clf2.predict(filtered)
    errors = abod_values[abod_values > 0]
    print "Total error (outliers not detected): {}%".format(float((len(errors))/float(len(ul))))
    print "{}/{} additional outliers have been detected".format(len(abod_values[abod_values < 0]), len(filtered))


def test_against_threshold():
    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_matthias_big.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb4 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # random.shuffle(emb1)
    random.shuffle(emb2)
    # random.shuffle(emb4)


    train = emb1[0:50]
    test = emb2[0:50]
    ul = emb4

    # ------ ABOD
    if False:

        clf = ABOD()
        clf.fit(train)
        pred_abod = clf.predict(ul)
        print "Misdetections ABOD (ul): {}".format(len(pred_abod[pred_abod > 0]))

        pred_abod = clf.predict(test)
        print "Misdetections ABOD (test): {}".format(len(pred_abod[pred_abod < 0]))

    # ------ THRESHOLDING

    t = BinaryThreshold()
    t.partial_fit(train)

    pred_thresh = t.predict(ul, True)

    print pred_thresh

    print "Misdetections Thresholding (ul): {}".format(len(pred_thresh[pred_thresh > 0]))





    print np.where(pred_thresh == False)[0]


    print np.nonzero(pred_thresh == 0)[0]

    pred_thresh = t.predict(test, True)




    print "Misdetections Thresholding (test): {}".format(len(pred_thresh[pred_thresh == 0]))




# ================================= #
#              Main

if __name__ == '__main__':
    # cascaded_classifiers()
    # test_ABOD()
    test_against_threshold()