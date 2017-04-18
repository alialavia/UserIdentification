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
from uids.v2.ClassifierController import IdentificationController, UpdateController


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

    is_save, samples = idc.try_to_identify(2, np.array([1,2,3,4,5,6]))
    print is_save, samples
    is_save, samples = idc.try_to_identify(2, np.array([7,8]))
    print is_save, samples
    # accumulation save now
    is_save, samples = idc.try_to_identify(2, np.array([9,10]))
    print is_save, samples
    # stack is deleted again
    is_save, samples = idc.try_to_identify(2, np.array([11, 12]))
    print is_save, samples


def test_data_controller():
    dc = DataController()
    dc.add_samples(1337, np.array([1,2,3]))
    cluster = dc.get_class_cluster(1337)
    dc.add_samples(1337, np.array([4,5,6]))
    print cluster.data


if __name__ == '__main__':
    # test_metrics()
    # test_identification_controller()

    test_data_controller()





