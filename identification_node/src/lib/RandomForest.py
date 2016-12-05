#!/usr/bin/env python2
# specific classifier
from sklearn.ensemble.forest import RandomForestClassifier as RF
import numpy as np
# classifier superclass
from src.lib.EmbeddingClassifier import EmbeddingClassifier

class RandomForest(EmbeddingClassifier):

    def __init__(self, user_db_):
        EmbeddingClassifier.__init__(self, user_db_)    # superclass init

    def define_classifier(self):
        self.classifier_tag = 'RandomForest'
        self.classifier = RF(n_estimators=10)
