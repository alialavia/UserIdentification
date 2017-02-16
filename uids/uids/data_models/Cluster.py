import numpy as np
import matplotlib.pyplot as plt
import time
from uids.utils.DataAnalysis import *
from scipy.spatial import Delaunay
from uids.utils.Logger import Logger as log
from uids.utils.KNFilter import KNFilter
from uids.data_models.ClusterBase import ClusterBase


class StandardCluster(ClusterBase):
    """
    Notes:
        - Convex Hull is a subgraph of the Delauny Triangulation
    """

    __max_size = None

    # ========= parameters

    def __init__(self, max_size=50):
        ClusterBase.__init__(self)
        self.__max_size = max_size

    def update(self, samples):
        


