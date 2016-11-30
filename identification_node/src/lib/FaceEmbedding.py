#!/usr/bin/env python2

import os
import time

import numpy as np
import pickle

import openface
import openface.helper
from openface.data import iterImgs

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')
classifierModelDir = os.path.join(modelDir, 'classification')

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


class FaceEmbedding:

    # settings
    dlibFacePredictor = "shape_predictor_68_face_landmarks.dat"
    landmarks = "outerEyesAndNose"
    size = 96
    skipMulti = True
    verbose = True
    networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')  # torch network model
    cuda = False

    neural_net = None       # torch network
    dlib_aligner = None     # dlib face aligner

    def __init__(self):

        start = time.time()
        print "--- loading models..."
        # load neural net
        self.neural_net = openface.TorchNeuralNet(self.networkModel, imgDim=self.size, cuda=self.cuda)
        # load dlib model
        self.dlib_aligner = openface.AlignDlib(dlibModelDir + "/" + self.dlibFacePredictor)
        print("--- model loading took {} seconds".format(time.time() - start))

    def get_embedding(self, user_img):
        args = Arguments()
        # align image
        normalized = self.align_face(user_img, args.landmarks, args.size)
        if normalized is None:
            return None

        # generate embedding
        rep = self.neural_net.forward(normalized)
        return rep

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