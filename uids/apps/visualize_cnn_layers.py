
import argparse
import os

import numpy as np

import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.spatial.distance import *
from sklearn.ensemble import GradientBoostingClassifier
# analysis tools
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn import datasets, cluster
from lib.DataAnalysis import *



from sklearn.cluster import FeatureAgglomeration


# path managing
file_dir__ = os.path.dirname(os.path.realpath(__file__))
model_dir__ = os.path.join(file_dir__, '..', 'models', 'embedding_samples')	# path to the model directory


# ================================= #
#              Utilities

def load_embeddings(filename):
    filename = "{}/{}".format(model_dir__, filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None

def dump_to_hd(filename, data):
    filename = "{}/{}".format(model_dir__, filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        f.close()

# ================================= #
#              Main


def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Image folder.", default="faces")
    parser.add_argument('--output', help="Statistics output folder.", default="stats")
    args = parser.parse_args()

    # load embeddings
    emb_1 = load_embeddings('embeddings_matthias.pkl')
    emb_2 = load_embeddings('embeddings_laia.pkl')
    emb_3 = load_embeddings('embeddings_elias.pkl')
    emb_lfw = load_embeddings('embeddings_lfw.pkl')

    # implementation
    # see: https://github.com/cmusatyalab/openface/blob/master/demos/vis-outputs.lua



if __name__ == '__main__':

    run_evaluation()
