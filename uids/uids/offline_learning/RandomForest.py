#!/usr/bin/env python2
from sklearn.ensemble.forest import RandomForestClassifier as RF
# classifier superclass
from uids.OfflineClassifierBase import OfflineClassifierBase


class RandomForest(OfflineClassifierBase):

    def __init__(self, user_db_):
        OfflineClassifierBase.__init__(self, user_db_)    # superclass init

    def define_classifier(self):
        self.classifier_tag = 'RandomForest'
        self.classifier = RF(n_estimators=10)
