import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
# classifiers
from sklearn.svm import SVC
import random

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory

from uids.offline_learning.SVM import SVM


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)
    if os.path.isfile(filename):
        # print filename
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return embeddings
    return None

#
# def test_sgd_incremental():
#
#     emb1 = load_embeddings("embeddings_matthias.pkl")
#     emb2 = load_embeddings("embeddings_lfw.pkl")
#     emb3 = load_embeddings("embeddings_laia.pkl")
#
#
#     # preprocess data ----------------
#
#     scaler = preprocessing.StandardScaler().fit(emb1)
#     emb1 = scaler.transform(emb1)
#     emb1_split = np.split(np.array(emb1), 2)
#     emb2 = scaler.transform(emb2)
#     emb3 = scaler.transform(emb3)
#     emb3_split = np.split(np.array(emb3), 2)
#
#     # -----------------
#
#     clf = linear_model.SGDClassifier()
#
#     all_classes = [12,3,2,1,5,6]
#     clf.partial_fit(emb1_split[0], np.repeat(1, len(emb1_split[0])), all_classes)
#     clf.partial_fit(emb2, np.repeat(2, len(emb2)))
#
#     pred = clf.predict(emb1_split[1])
#     print "{}/{} inliers have been detected".format(len(pred[pred==1]), len(pred))
#
#     # predict novelty
#     pred = clf.predict(emb3)
#     print "{}/{} inliers have been detected".format(len(pred[pred==1]), len(pred))
#
#
#     # train novelty
#     clf.partial_fit(emb3_split[0], np.repeat(3, len(emb3_split[0])))
#
#     # predict class
#     pred = clf.predict(emb3_split[1])
#     print "{}/{} inliers have been detected".format(len(pred[pred==3]), len(emb3_split[1]))


# ================================= #
#              Main

def test1():
    emb1 = load_embeddings("embeddings_matthias.pkl")
    emb2 = load_embeddings("embeddings_matthias_big.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb4 = load_embeddings("embeddings_christian.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    random.shuffle(emb1)

    clf = SVC(kernel='linear', probability=True, C=1)

    train = emb1[0:30]
    test = emb2
    ul = emb4

    # train user and unknown class
    label_class = np.repeat(1, np.shape(train)[0])
    label_unknown = np.repeat(0, np.shape(emb_lfw)[0])
    training_embeddings = np.concatenate((train, emb_lfw))
    training_labels = np.concatenate((label_class, label_unknown))
    clf.fit(training_embeddings, training_labels)

    # ---------------------- 1. test on training class
    prediction = clf.predict(test)
    errors = len(test)-np.sum(prediction)
    print "Inlier Recognition - Error rate: {:.2f}%".format(float(errors)/len(test)*100.0)
    class_mask = prediction == 1
    novelty_mask = prediction == 0
    proba = clf.predict_proba(test)

    # samples that identify class
    pos_class_proba = proba[class_mask, :]
    inlier_proba = pos_class_proba[:,1]

    # samples that identify novelty
    pos_novelty_proba = proba[novelty_mask, :]
    outlier_proba = pos_novelty_proba[:,1]

    # min/max confidence
    print "Is an outlier probability: min {:.2f} - max {:.2f}".format(np.min(outlier_proba), np.max(outlier_proba))
    # apply threshold
    nr_uncertain_samples = len(outlier_proba[outlier_proba < 0.7])
    print "{}/{} of the misclassified samples are uncertain (prob. < 70%)".format(nr_uncertain_samples, errors)
    print "-----------------------------------------------"
    print "Inlier Recognition - Error rate (dropping uncertain samples): {:.2f}%".format(float(errors-nr_uncertain_samples)/(len(test)-nr_uncertain_samples)*100.0)
    print "-----------------------------------------------"
    # ---------------------- 2. test on novel class

    prediction = clf.predict(ul)
    errors = np.sum(prediction)
    print "Novelty Recognition - Error rate: {:.2f}%".format(float(errors)/len(ul)*100.0)

    class_mask = prediction == 1
    novelty_mask = prediction == 0
    proba = clf.predict_proba(ul)

    # samples that identify novelty
    if errors > 0:
        pos_novelty_proba = proba[novelty_mask, :]
        outlier_proba = pos_novelty_proba[:, 1]
    else:
        outlier_proba = proba[:,0]

    # min/max confidence
    print "Is an outlier probability: min {:.2f} - max {:.2f}".format(np.min(outlier_proba), np.max(outlier_proba))
    # apply threshold
    nr_uncertain_samples = len(outlier_proba[outlier_proba < 0.7])
    print "{}/{} of the (as outliers) classified samples are uncertain (prob. < 70%)".format(nr_uncertain_samples, len(ul)-errors)
    # print "Inlier Recognition - Error rate (dropping uncertain samples): {:.2f}%".format(float(errors-nr_uncertain_samples)/(len(test)-nr_uncertain_samples)*100.0)


if __name__ == '__main__':
    test1()