import numpy as np
import Queue
from sklearn.ensemble import IsolationForest
from uids.utils.Logger import Logger as log
# v2 models
from uids.v2.set_metrics import ABOD
from uids.v2.HardThreshold import SetSimilarityHardThreshold
from uids.data_models.StandardCluster import StandardCluster
from uids.v2.MultiClassClassifierBase import MultiClassClassifierBase
from uids.v2.DataController import DataController
from uids.v2.ClassifierController import IdentificationController, UpdateController, BaseMetaController
from uids.v2.set_metrics import *
import os
import random
import pickle

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')  # path to the model directory


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None


def test_metrics():
    cls = SetSimilarityHardThreshold(metric='ABOD', threshold=0.7)
    cls.partial_fit([[2, 1], [4, 65], [4, 3]])
    dec, scores = cls.predict([[2.2, 1], [4, 1]])
    print dec, scores


def test_external_cluster():

    ext_data = StandardCluster()
    ext_data.update([[2, 1], [4, 65], [4, 3]])
    cls = SetSimilarityHardThreshold(metric='ABOD', threshold=0.7, cluster=ext_data)
    dec, scores = cls.predict([[2.2, 1], [4, 1]])
    print dec, scores


def test_queue_block():
    q = Queue.Queue(maxsize=0)
    q.put(1)
    q.put(2)

    while True:
        try:
            task = q.get(False)
        except Queue.Empty:
            # Handle empty queue here
            print "nooo"
            pass
            # check timeout trainings
        else:
            # Handle task here and call q.task_done()
            print "processing manual task"
            print task
            q.task_done()


def test_identification_controller():

    idc = IdentificationController({})

    is_save, samples = idc.accumulate_samples(2, np.array([1, 2, 3, 4, 5, 6]))
    print is_save, samples
    is_save, samples = idc.accumulate_samples(2, np.array([7, 8]))
    print is_save, samples
    # accumulation save now
    is_save, samples = idc.accumulate_samples(2, np.array([9, 10]))
    print is_save, samples
    # stack is deleted again
    is_save, samples = idc.accumulate_samples(2, np.array([11, 12]))
    print is_save, samples


def test_data_controller():
    dc = DataController()
    dc.add_samples(1337, np.array([1,2,3]))
    cluster = dc.get_class_cluster(1337)
    dc.add_samples(1337, np.array([4,5,6]))
    print cluster.data


def test_abod():

    emb2 = load_embeddings("matthias_test2.pkl")
    emb3 = load_embeddings("embeddings_christian_clean.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")


    test_with_ul = np.concatenate((emb2[1:3], emb3[0:10]))
    random.shuffle(test_with_ul)


    # print BaseMetaController.check_inter_sample_dist(test_with_ul)
    print BaseMetaController.check_inter_sample_dist(emb3)
    print BaseMetaController.check_inter_sample_dist(emb_lfw[0:1000])
    # print ABOD.get_set_score(emb3[0:4])
    # print ABOD.get_set_score(emb_lfw[0:4])

def test_normed_confidence():

    weights = np.array([2,6,9])
    predictions = [0,0,1]
    norm_f = 1.0/np.sum(weights)

    confidence = np.dot(predictions, np.transpose(norm_f * weights))
    print confidence


def test_arr_append():

    l = []


    l.append(2)
    l.append(4.2)

    l.append(np.array([3.4]))
    l.append(np.array([8.745]))


    print np.array(l).flatten()


    t = np.array([8.745, 9.234])
    print t
    print np.array(t).flatten()


if __name__ == '__main__':
    # test_metrics()
    # test_identification_controller()

    # test_normed_confidence()
    test_arr_append()


