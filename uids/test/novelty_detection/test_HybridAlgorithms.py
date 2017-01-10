from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
import os
import pickle
import numpy as np

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
        return np.array(embeddings)
    return None


# ================================= #
#        Test Functions

def test_1():
    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    # prepare ds
    np.random.shuffle(emb2)
    if len(emb2) % 2 != 0:
        emb2 = emb2[:-1]

    split_set = np.array_split(emb2, 2)

    X_train = split_set[0]
    X_test = split_set[1]
    X_outliers = emb_lfw


    K = chi2_kernel(X_train, gamma=.5)
    print K

    # svm = SVC(kernel='precomputed').fit(K, y)
    # svm.predict(K)


# ================================= #
#              Main

if __name__ == '__main__':
    test_1()
