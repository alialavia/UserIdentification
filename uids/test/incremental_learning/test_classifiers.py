import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import pickle
import os
from sklearn import svm
from sklearn import linear_model

from sklearn.linear_model import SGDClassifier
import random

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory

def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return embeddings
    return None


def train_and_classify(class_data, outlier_data, clf):

    print "=====================================================\n"
    print "        EVALUATE CLASSIFIER\n"
    print "=====================================================\n"

    all_classes = [12,3,2,1,5,6]
    labels = np.repeat(2, len(class_data))
    start = time.time()
    clf.partial_fit(class_data, labels, all_classes)
    print "fitting took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_1 = clf.predict(class_data)
    print "prediction took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_2 = clf.predict(outlier_data)
    print "prediction took {} seconds".format(time.time()-start)

    print "{}/{} outliers have been detected".format((label_pred_2 < 0).sum(), len(outlier_data))
    print "{}/{} inliers have been detected".format((label_pred_1 > 0).sum(), len(class_data))


def batches(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def test_sgd(X, Y):
    from sklearn.linear_model import SGDClassifier
    import random
    clf2 = SGDClassifier(loss='log')  # shuffle=True is useless here
    shuffledRange = range(len(X))
    n_iter = 5
    for n in range(n_iter):
        random.shuffle(shuffledRange)
        shuffledX = [X[i] for i in shuffledRange]
        shuffledY = [Y[i] for i in shuffledRange]
        for batch in batches(range(len(shuffledX)), 10000):
            clf2.partial_fit(shuffledX[batch[0]:batch[-1] + 1], shuffledY[batch[0]:batch[-1] + 1],
                             classes=np.unique(Y))


# ================================= #
#              Main

if __name__ == '__main__':

    filename = "embeddings_elias.pkl"
    emb1 = load_embeddings(filename)
    filename = "embeddings_lfw.pkl"
    emb2 = load_embeddings(filename)

    sgda = linear_model.SGDClassifier(average=True)


    train_and_classify(emb1, emb2, sgda)






