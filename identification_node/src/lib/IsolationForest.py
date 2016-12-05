#!/usr/bin/env python2
# specific classifier
from sklearn.ensemble import IsolationForest as IF
import numpy as np
# classifier superclass
from src.lib.EmbeddingClassifier import EmbeddingClassifier

class IsolationForest(EmbeddingClassifier):

    def __init__(self, user_db_):
        EmbeddingClassifier.__init__(self, user_db_)    # superclass init

    def define_classifier(self):
        self.classifier_tag = 'IsolationForest'
        self.classifier = IF(n_estimators=10)
