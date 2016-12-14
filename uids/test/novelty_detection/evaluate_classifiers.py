import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import pickle
import os
from sklearn import svm
from sklearn import linear_model
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


def train_on_single_ds_and_classify(class_data, random_data, clf):

    print "=====================================================\n"
    print "        EVALUATE CLASSIFIER\n"
    print "=====================================================\n"

    start = time.time()
    clf.fit(class_data)
    print "fitting took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_1 = clf.predict(class_data)
    print "prediction took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_2 = clf.predict(random_data)
    print "prediction took {} seconds".format(time.time()-start)

    print "{}/{} outliers have been detected".format((label_pred_2 < 0).sum(), len(random_data))
    print "{}/{} inliers have been detected".format((label_pred_1 > 0).sum(), len(class_data))





# ================================= #
#              Main

if __name__ == '__main__':

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    emb1 = np.concatenate((emb1,emb2,emb3,emb1,emb3))

    print np.shape(emb1)

    rf = IsolationForest(random_state=np.random.RandomState(42))
    ocsvm = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
    ocsvm2 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

    train_on_single_ds_and_classify(emb1, emb2, rf)
    train_on_single_ds_and_classify(emb1, emb2, ocsvm)
    train_on_single_ds_and_classify(emb1, emb2, ocsvm2)

