from uids.online_learning.ABOD import ABOD
from uids.utils.Logger import Logger as log
from uids.utils.HullCluster import HullCluster


class IABOD(ABOD):

    data = []
    __verbose = False
    data_cluster = HullCluster()

    def __init__(self):
        ABOD.__init__(self)

    def partial_fit(self, samples):
        if len(self.data) == 0:
            # init on first call
            self.fit(samples)
        self.data_cluster.update(samples)
        self.data = self.data_cluster.get_data()
        log.severe("IABOD cluster: {}".format(len(self.data)))

