#!/usr/bin/env python2
# specific classifier
from sklearn.ensemble.forest import RandomForestClassifier as RF
import numpy as np
# classifier superclass
from uids.lib.OfflineLearning.OfflineClassifierBase import OfflineClassifierBase

class RandomForest(OfflineClassifierBase):

    def __init__(self, user_db_):
        OfflineClassifierBase.__init__(self, user_db_)    # superclass init

    def define_classifier(self):
        self.classifier_tag = 'RandomForest'
        self.classifier = RF(n_estimators=10)
