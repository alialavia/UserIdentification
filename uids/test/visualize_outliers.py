#!/usr/bin/env python2
import argparse
import os
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
import math
# analysis tools
from uids.utils.DataAnalysis import *
from sklearn import metrics
from sklearn.metrics.cluster import *
from external.jqmcvi.base import *
import time
import sys
import csv
from matplotlib.pyplot import imshow
from scipy import misc

from sklearn.metrics.pairwise import *
from PIL import Image
from numpy import genfromtxt

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models', 'embedding_samples')	# path to the model directory
ressourceDir = os.path.join(fileDir, '..', 'ressource')	# path to the ressource directory


def load_data(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
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


def load_log(filename):
    filename = "{}/{}".format(modelDir, filename)
    # print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            content = np.array(list(reader))

            print(content[0,:])

            # for row in reader:
            #     print(', '.join(row))
    return None


class LogFile:

    data = None
    filename = ""

    def __init__(self):
        pass

    def load_log(self, filename):
        filename = "{}/{}".format(modelDir, filename)
        # print filename
        if os.path.isfile(filename):
            self.filename = filename
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                self.data = np.array(list(reader))

    def save_log(self):
        with open(self.filename, 'wb') as f:
            writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for r in self.data:
                writer.writerows(r)
            f.close()

    def get_log(self, picture_index):
        return self.data[picture_index, :]

    def clean_log(self, picture_dir):

        path = os.path.join(ressourceDir, picture_dir)
        clean = []

        if os.path.isdir(path):
            self.dir = path
            for i, f in enumerate(self.data[:, 0]):
                filename = "{}/{}".format(path, f)
                if os.path.isfile(filename):
                    clean.append(self.data[i,:])
        else:
            print "Picture path {} does not exist".format(path)
            return

        print "Log size reduzed to {}/{}".format(len(clean), len(self.data))
        self.data = clean

    def get_picture_names(self, mask):

        d =np.array(self.data)
        extr = d[mask]
        return extr[:,0]

    def get_picture_name(self, picture_index):
        return self.data[picture_index, 0]

    def get_roll(self, picture_index):
        return self.data[picture_index, 1]

    def get_pitch(self, picture_index):
        return self.data[picture_index, 2]

    def get_yaw(self, picture_index):
        return self.data[picture_index, 3]


class PictureLoader:

    images = []
    dir = None

    def load_pictures(self, filenames, dirname):
        self.images[:] = []
        self.images = []

        path = os.path.join(ressourceDir, dirname)

        if os.path.isdir(path):
            self.dir = path
            for i, f in enumerate(filenames):
                filename = "{}/{}".format(path, f)
                image = misc.imread(filename)
                self.images.append(image)
        else:
            print "Picture path {} does not exist".format(path)
            return

    def get_picture(self, index):
        if not self.images:
            return None
        return self.images[index]

    def get_picture_matrix(self):

        # for i in self.images:
        #     img = Image.fromarray(i, 'RGB')
        #     Image._show(img)
        #     # Image.show(i)
        #     # misc.pilutil.imshow(i)


        pil_images = []
        for i in self.images:
            pil_images.append(Image.fromarray(i, 'RGB'))

        widths, heights = zip(*(i.size for i in pil_images))
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in pil_images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        # use pyplot for visualization
        imshow(np.asarray(new_im))
        plt.show()

        # new_im.show()

# ================================= #
#              Main


def run_evaluation():

    # 0: no blur, 1: 1.5, 2: 3, 4: 4.5
    emb1 = load_data("embedding_frontal_increasing_blur.pkl")
    emb2 = load_data("embeddings_matthias_big.pkl")

    pl = PictureLoader()
    log = LogFile()
    log.load_log('picture_log_matthias_big.csv')

    log.clean_log('matthias_big')
    # log.save_log()


    img_loader = PictureLoader()

    # # filter blurred images
    # l = load_labels('blur_labels_matthias_big.csv')
    # l = l[:,1]
    # blurred = emb2[l==1]
    # clear = emb2[l==0]

    ref_face = emb1[0,:]

    # calculate intra class distribution
    sep = pairwise_distances(ref_face, emb2, metric='cosine')
    n, bins, patches = plt.hist(np.transpose(sep), 50, normed=1, facecolor='green', alpha=0.75)
    plt.show()

    # apply threshold
    threshold = 0.45
    outlier_mask = sep > threshold
    best_inlier_mask = sep < 0.1

    # display outliers
    outlier_filenames = log.get_picture_names(outlier_mask[0])
    pl.load_pictures(outlier_filenames, 'matthias_big_aligned')
    pl.get_picture_matrix()

    # display inliers
    inlier_filenames = log.get_picture_names(best_inlier_mask[0])
    pl.load_pictures(inlier_filenames, 'matthias_big_aligned')
    pl.get_picture_matrix()

    # display outliers


if __name__ == '__main__':
    run_evaluation()