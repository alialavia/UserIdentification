import math

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances
import time
import matplotlib.pyplot as plt
import pylab
from sklearn.datasets.samples_generator import make_blobs

class Cluster:

    centroid = None
    total_points = 0

    # initial data
    init_data = None

    def __init__(self, init_data):
        self.init_data = init_data

    def calc_centroid(self, data):
        embedding = zip(*data)
        num_points = len(data)
        coords = [math.fsum(dim_list) / num_points for dim_list in embedding]
        return coords

    def init(self, features):
        # calculate cluster centroid
        self.init_data = features
        self.total_points = len(features)
        self.centroid = self.calc_centroid(self.init_data)

    def update(self, feature):
        self.total_points = self.total_points + 1


class CNN:
    shrink_threshold_ = False

    # initialization data
    init_data_ = None
    init_labels_ = None

    # labels
    label_encoder_ = None    # classifier label encoder


    # centroids
    centroids_ = None
    nr_centroid_points_ = []
    classes_ = None  # list of original classes

    # status
    initialized_ = False
    updates_ = 0

    # settings
    metric_ = 'mahalanobis'

    def update_centroid_single(self, point, cent_class):
        icl = self.label_encoder_.transform(cent_class)
        nr_pts = self.nr_centroid_points_[icl]
        old_cent = self.nr_centroid_points_[icl]
        self.nr_centroid_points_[icl] = (point+nr_pts*old_cent)/(nr_pts+1)
        self.nr_centroid_points_[icl] = nr_pts + 1

    def __init__(self, init_data, init_labels):
        self.init_data_ = init_data
        self.init_labels_ = init_labels

        n_samples, n_features = init_data.shape # 2 x 6

        # initialize label encoder
        self.label_encoder_ = LabelEncoder()
        labels_numeric_ = self.label_encoder_.fit_transform(init_labels)


        # get original classes
        self.classes_ = self.label_encoder_.classes_


        n_classes = self.classes_.size

        if n_classes < 2:
            raise ValueError('Initialization set needs at least 2 classes')

        # empty centroid for each class and each dimension
        self.centroids_ = np.empty((n_classes, n_features), dtype=np.float64)

        # Number of points in each class
        nk = np.zeros(n_classes)

        # print nk

        for cur_class in range(n_classes):
            center_mask = labels_numeric_ == cur_class

            # calculate number of points in each class
            nk[cur_class] = np.sum(center_mask)

            # calculate euclidean mean/centroid
            self.centroids_[cur_class] = init_data[center_mask].mean(axis=0)


        #print self.centroids_

        if self.shrink_threshold_:
            dataset_centroid_ = np.mean(X, axis=0)

            # m parameter for determining deviation
            m = np.sqrt((1. / nk) + (1. / n_samples))
            # Calculate deviation using the standard deviation of centroids.
            variance = (X - self.centroids_[y_ind]) ** 2
            variance = variance.sum(axis=0)
            s = np.sqrt(variance / (n_samples - n_classes))
            s += np.median(s)  # To deter outliers from affecting the results.
            mm = m.reshape(len(m), 1)  # Reshape to allow broadcasting.
            ms = mm * s
            deviation = ((self.centroids_ - dataset_centroid_) / ms)
            # Soft thresholding: if the deviation crosses 0 during shrinking,
            # it becomes zero.
            signs = np.sign(deviation)
            deviation = (np.abs(deviation) - self.shrink_threshold)
            deviation[deviation < 0] = 0
            deviation *= signs
            # Now adjust the centroids using the deviation
            msd = ms * deviation

            self.centroids_ = dataset_centroid_[np.newaxis, :] + msd

        initialized_ = True

    def predict(self, feature):
        distances = pairwise_distances( feature, self.centroids_, metric=self.metric_)
        min_dist = distances.argmin(axis=1)

        print self.classes_

        #print self.centroids_
        #print min_dist
        return self.classes_[min_dist]


if __name__ == '__main__':

    # ----------- generate data
    np.random.seed(0)
    batch_size = 45
    centers = [[2, 1], [-1, -1], [1, -1]]
    n_clusters = len(centers)
    X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.3)

    labels_true[labels_true==1] = 3
    # -------------

    start = time.time()
    classifier = CNN(X,labels_true)
    print "--- Initialization/training took {} seconds".format(time.time()-start)
    # ------------------

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#009933', '#ff9900', '#ff0066', '#EA5A0C', '#33cccc']

    # KMeans
    ax = fig.add_subplot(1, 2, 1)
    for k, col in zip(range(4), colors):
        my_members = labels_true == k
        #cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='o')
        #ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #        markeredgecolor='k', markersize=6)

    #plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (
    #t_batch, k_means.inertia_))

    ax.set_title('Original data')
    ax.set_xticks(())
    ax.set_yticks(())

    # predict
    points = [[-0.8, -1], [1,-0.9], [0.001, -1], [-0.001, -1], [0.1, -1], [0.2, -1], [0.4, -1]]


    points.append([2,0.9])
    ax = fig.add_subplot(1, 2, 2)

    for k, col in zip(range(4), colors):
        my_members = labels_true == k
        #cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor='#D3D3D3', marker='o')

    for i, pt in zip(range(len(points)), points):
        label = classifier.predict([pt])
        ax.plot(pt[0], pt[1], 'w',
                markerfacecolor=colors[label], marker='o')


    ax.set_title('New Points')
    ax.set_xticks(())
    ax.set_yticks(())

    plt.show()
