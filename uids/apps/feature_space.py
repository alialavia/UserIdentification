
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

class PlotHandler:

    __fig_count = 1
    __title = ""

    def Show(self):
        plt.show()

    def SetTitle(self, title):
        self.__title = title

    def PlotVarianceContribution(self, embeddings):
        """Plot explained data Variance vs Vector dimensions"""

        var_ratio = CalcComponentVariance(embeddings)
        vector_dims = np.size(embeddings,1)
        vector_range = np.linspace(0, vector_dims-1, num=vector_dims)
        data = []
        for i_lim in vector_range:
            data.append(np.sum(var_ratio[0:int(i_lim)])*100)

        fig = plt.figure(self.__fig_count)
        self.__fig_count = self.__fig_count + 1

        # ax = fig.add_subplot(111)
        plt.plot(vector_range, data)

        plt.ylim([0, 100])
        plt.xlim([0, vector_dims])
        if self.__title is not "":
            plt.title(self.__title)
        else:
            plt.title('Explained Data Variance')
        plt.ylabel('Explained variance [%]')
        plt.xlabel('Number of features')
        __title = ""


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

    if emb_1 is None or emb_2 is None:
        print "--- embeddings could not be loaded. Aborting..."
        return

    # ------------------- EVALUATION ON ORIGINAL VECTORS



    ph = PlotHandler()

    # ==== 1. PCA DIMENSION REDUCTION

    # ph.PlotVarianceContribution(emb_lfw)
    # # reduce dimensionality
    # basis, mean = ExtractSubspace(emb_lfw, 0.999)
    # # dump_to_hd("lfw_99.9_subspace.pkl", (basis, mean))
    #
    # reduced_data = ProjectOntoSubspace(emb_lfw, mean, basis)
    # ph.SetTitle("Component Variance Contribution on Subspace")
    # ph.PlotVarianceContribution(reduced_data)
    # ph.Show()


    # ==== 1. FEATURE AGGLOMERATION

    agglo = FeatureAgglomeration(n_clusters=20)
    agglo.fit(emb_lfw)
    X_reduced = agglo.transform(emb_1)

    print np.shape(X_reduced)

    # ==== 1. PCA DIMENSION REDUCTION


    # labels = np.concatenate((np.repeat(1, len(emb_1)), np.repeat(2, len(emb_2))))
    # data = np.concatenate((emb_1, emb_2))
    #
    # clf = GradientBoostingClassifier().fit(data, labels)
    # features = [3, 100]
    # fig, axs = plot_partial_dependence(clf, data, features, label=0)


if __name__ == '__main__':

    run_evaluation()
