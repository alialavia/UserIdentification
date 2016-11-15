#!/usr/bin/env python2

import os
import time

import numpy as np

import openface
import openface.helper
from openface.data import iterImgs

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')

# classifiers
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# argument container
# TODO: refactor this properly!
class Arguments:
    def __init__(self):
        self.dlibFacePredictor = "shape_predictor_68_face_landmarks.dat"
        self.landmarks = "outerEyesAndNose"
        self.size = 96
        self.skipMulti = True
        self.verbose = True
        # embedding calculation
        self.networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7') # torch network model
        self.cuda = False


class OfflineUserClassifier:
    # key: user id, value: list of embeddings
    user_embeddings = {}    # raw embeddings
    neural_net = None       # torch network
    dlib_aligner = None     # dlib face aligner
    classifier = None              # classifier
    label_encoder = None    # classifier label encoder

    def __init__(self):
        args = Arguments()

        start = time.time()
        print "--- loading models..."
        # load neural net
        self.neural_net = openface.TorchNeuralNet(args.networkModel, imgDim=args.size, cuda=args.cuda)
        # load dlib model
        self.dlib_aligner = openface.AlignDlib(dlibModelDir + "/" + args.dlibFacePredictor)
        # initialize classifier
        self.classifier = SVC(C=1, kernel='linear', probability=True)

        print("--- identifier initialization took {} seconds".format(time.time() - start))

    def collect_embeddings(self, images, user_id):
        """collect embeddings of faces to train detect - for a single user id"""

        args = Arguments()

        print "--- Starting normalization..."
        # normalize images
        images_normalized = []
        start = time.time()
        if len(images) > 0:
            for imgObject in images:
                # align face - ignore images with multiple bounding boxes
                aligned = self.align_face(imgObject, args.landmarks, args.size)
                if aligned is not None:
                    images_normalized.append(aligned)

        if len(images_normalized) > 0:
            print("--- Alignment took {} seconds - " + str(len(images_normalized)) + "/" + str(len(images)) + " images suitable".format(time.time() - start))

        else:
            print "--- No suitable images (no faces detected)"

        # generate embeddings
        reps = []
        for img in images_normalized:
            start = time.time()
            rep = self.neural_net.forward(img)
            print("--- = Neural network forward pass took {} seconds.".format(
                time.time() - start))
            reps.append(rep)

        # save
        if user_id in self.user_embeddings:
            # append
            print "--- Appending "+str(len(reps))+" embeddings"
            self.user_embeddings[user_id].append(reps)
        else:
            self.user_embeddings[user_id] = reps

        # display current representations
        self.print_embedding_status()

    def identify_user(self, user_img):

        start = time.time()
        embedding = self.get_embedding(user_img)

        if embedding is None:
            return None

        embedding = embedding.reshape(1, -1)

        # alternative - predicts index of label array
        # user_id = self.classifier.predict(embedding)

        # prediction probabilities
        probabilities = self.classifier.predict_proba(embedding).ravel()
        maxI = np.argmax(probabilities)
        confidence = probabilities[maxI]
        user_id_pred = self.label_encoder.inverse_transform(maxI)
        print("--- Identification took {} seconds.".format(time.time() - start))

        return (user_id_pred, confidence)

    #  ----------- UTILITIES

    def get_embedding(self, user_img):
        args = Arguments()
        # align image
        normalized = self.align_face(user_img, args.landmarks, args.size)
        if normalized is None:
            return None

        # generate embedding
        rep = self.neural_net.forward(normalized)
        return rep

    def print_embedding_status(self):
        print "--- Current embeddings:"
        for user_id, embeddings in self.user_embeddings.iteritems():
            print "     User" + str(user_id) + ": " + str(len(embeddings)) + " representations"

    def trigger_training(self):
        """triggers the detector training from the collected faces"""
        print "--- Triggered classifier training"

        if len(self.user_embeddings) < 2:
            print "Number of users must be greater than one"
            return

        start = time.time()
        embeddings_accumulated = []
        labels = []
        for user_id, user_embeddings in self.user_embeddings.iteritems():
            # add label
            labels = np.append(labels, np.repeat(user_id, len(user_embeddings)))
            print(user_embeddings)
            if not embeddings_accumulated:
                embeddings_accumulated = user_embeddings
            else:
                embeddings_accumulated = np.concatenate((embeddings_accumulated, user_embeddings))

        embeddings_accumulated = np.array(embeddings_accumulated)

        self.label_encoder = LabelEncoder().fit(labels)
        labelsNum = self.label_encoder.transform(labels)

        # train classifier
        self.classifier.fit(embeddings_accumulated, labelsNum)
        print("    Classifier training took %s seconds for "+str(embeddings_accumulated.shape[0])+" embeddings." % time.time() - start)

    def align_face(self, image, landmark, output_size, skip_multi=False):

        landmarkMap = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if landmark not in landmarkMap:
            raise Exception("Landmarks unrecognized: {}".format(landmark))

        landmarkIndices = landmarkMap[landmark]

        # TODO: check if is really output size or input size
        # align image
        outRgb = self.dlib_aligner.align(output_size, image,
                             landmarkIndices=landmarkIndices,
                             skipMulti=skip_multi)
        if outRgb is None:
            print("--- Unable to align.")

        return outRgb