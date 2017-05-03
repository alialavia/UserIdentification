import numpy as np
from uids.utils.Logger import Logger as log
from uids.data_models.MeanShiftCluster import MeanShiftCluster, MeanShiftPoseCluster
from uids.features.ConfidenceGen import WeightGenerator


class DataController:

    # raw CNN embeddings in clusters
    class_clusters = {}
    weight_gen = None

    def __init__(self):
        self.weight_gen = WeightGenerator()

    # --------- CLASS HASHING

    def classes_in_range(self, samples, thresh=1.3, metric='euclidean'):
        class_ids = []
        for id, c in self.class_clusters.iteritems():
            # only predict for "reasonable"/near classes
            range = np.mean(c.class_mean_dist(samples, metric))
            if range < thresh:
                class_ids.append(id)
            else:
                log.info('db', "Class {} out of range [{}] ({} [ref] < {:.3f})".format(id, metric, thresh, range))

        return class_ids

    def get_class_means(self):
        return [c.mean() for id, c in self.class_clusters.iteritems()]

    # --------- DATA MANAGEMENT (regular clusters)

    def add_samples(self, user_id, new_samples):
        """embeddings: array of embeddings"""
        if user_id not in self.class_clusters:
            # initialize
            self.class_clusters[user_id] = MeanShiftCluster(max_size=60)
            self.class_clusters[user_id].update(new_samples)
        else:
            # update
            self.class_clusters[user_id].update(new_samples)

    def get_class_samples(self, class_id):
        if class_id in self.class_clusters:
            return self.class_clusters[class_id].get_data()
        else:
            return None

    def get_class_cluster(self, class_id):
        if class_id in self.class_clusters:
            return self.class_clusters[class_id]
        else:
            return None

    def class_mean_distances(self, samples, class_ids, metric='euclidean'):
        distances = []
        class_ids_clean = []
        for class_id, __clf in self.class_clusters.iteritems():
            if class_id in class_ids:
                class_ids_clean.append(class_id)
                distances.append(np.mean(self.class_clusters[class_id].class_mean_dist(samples, metric)))
        return np.array(distances), np.array(class_ids_clean)

    # --------- MANIFOLD LEARNING

    # TODO: implement
    def merge_near_classes(self):
        pass
        # for id, c in self.class_clusters.iteritems():
        #     c.mean()


