import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import pickle
import os

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'src')	# path to the model directory

def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return embeddings
    return None


# ================================= #
#              Main

if __name__ == '__main__':

    filename = "embeddings_matthias.pkl"
    embeddings_1 = load_embeddings(filename)
    filename = "embeddings_christian.pkl"
    embeddings_2 = load_embeddings(filename)


    # fit the model
    rng = np.random.RandomState(42)
    clf = IsolationForest(random_state=rng)
    start = time.time()
    clf.fit(embeddings_1)
    print "fitting took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_1 = clf.predict(embeddings_1)
    print "prediction took {} seconds".format(time.time()-start)

    start = time.time()
    label_pred_2 = clf.predict(embeddings_2)
    print "prediction took {} seconds".format(time.time()-start)

    print "{}/{} outliers have been detected".format((label_pred_2 < 0).sum(), len(embeddings_2))
    print "{}/{} inliers have been detected".format((label_pred_1 > 0).sum(), len(embeddings_1))
