#!/usr/bin/env python2

import os
import time
import pickle
from abc import abstractmethod
# classifiers
from sklearn.preprocessing import LabelEncoder

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')
classifierModelDir = os.path.join(modelDir, 'classification')


class EmbeddingClassifier:

    # link to user database
    __p_user_db = None

    classifier = None       # classifier
    classifier_tag = None   # classifier short name (for storage purposes)
    label_encoder = None    # classifier label encoder
    training_status = False

    def __init__(self, user_db_):

        # link database
        __p_user_db = user_db_

        start = time.time()
        print "--- loading models..."

        # initialize classifier
        self.define_classifier()
        if self.classifier is None or self.classifier_tag is None:
            raise NotImplementedError("The initialization method for the classifier is not properly implemented.")

        # load stored classifier
        self.training_status = self.load_classifier()
        print("--- classifier initialization took {} seconds".format(time.time() - start))

    @abstractmethod
    def define_classifier(self):
        """Initialize the classifier"""
        # e.g.  self.classifier = SVC(C=1, kernel='linear', probability=True)
        raise NotImplementedError( "The initialization method for the classifier must be implemented." )

    def predict_label(self, user_embedding):
        if self.training_status is False:
            return None

        user_embedding = user_embedding.reshape(1, -1)
        start = time.time()
        label_encoded = self.classifier.predict(user_embedding)
        label_predicted = self.label_encoder.inverse_transform(label_encoded)
        print("--- Identification took {} seconds.".format(time.time() - start))
        return label_predicted

    #  ----------- UTILITIES

    def load_classifier(self):
        filename = "{}/"+self.classifier_tag+"_classifier.pkl".format(classifierModelDir)
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                (self.id_increment, self.user_list, self.user_embeddings, self.label_encoder, self.classifier) = pickle.load(f)
            return True
        return False

    def store_classifier(self):
        if self.training_status is False:
            print("--- Classifier is not trained yet")
            return

        filename = "{}/"+self.classifier_tag+"_classifier.pkl".format(classifierModelDir)
        print("--- Saving classifier to '{}'".format(filename))
        with open(filename, 'wb') as f:
            pickle.dump((self.id_increment, self.user_list, self.user_embeddings, self.label_encoder, self.classifier), f)

    def trigger_training(self):
        """triggers the detector training from the collected faces"""
        print "--- Triggered classifier training"

        if len(self.__p_user_db.user_embeddings) < 2:
            print "--- Number of users must be greater than one. Trying to load stored model..."
            if self.load_classifier() is True:
                print "--- Classifier loaded from file."
            else:
                print "--- Could not find classifier model."
            return

        start = time.time()

        (embeddings, labels) = self.__p_user_db.get_labeled_embeddings()

        # transform to numerical labels
        self.label_encoder = LabelEncoder().fit(labels)

        # get numerical labels
        labels_numeric = self.label_encoder.transform(labels)

        # train classifier
        self.classifier.fit(embeddings, labels_numeric)
        print("    Classifier training took {} seconds".format(time.time() - start))
        self.training_status = True

        # store classifier
        self.store_classifier()