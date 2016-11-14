#!/usr/bin/env python2

import os
import time

import openface
import openface.helper
from openface.data import iterImgs

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')

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
    user_embeddings = {}
    neural_net = None
    dlib_aligner = None

    def __init__(self):
        args = Arguments()

        start = time.time()
        print "--- loading models..."
        # load neural net
        self.neural_net = openface.TorchNeuralNet(args.networkModel, imgDim=args.size, cuda=args.cuda)
        # load dlib model
        self.dlib_aligner = openface.AlignDlib(dlibModelDir + "/" + args.dlibFacePredictor)
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
                aligned = self.align_face(args, imgObject)
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
            self.user_embeddings[user_id].append(reps)
        else:
            self.user_embeddings[user_id] = reps

        # display current representations
        self.print_embedding_status()

    #  ----------- UTILITIES

    def print_embedding_status(self):
        print "--- Current embeddings:"
        for user_id, embeddings in self.user_embeddings.iteritems():
            print "     User" + str(user_id) + ": " + str(len(embeddings)) + " representations"

    def trigger_training(self):
        """triggers the detector training from the collected faces"""
        print "--- trigger training"

    def align_face(self, args, image, multiple=False):

        landmarkMap = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if args.landmarks not in landmarkMap:
            raise Exception("Landmarks unrecognized: {}".format(args.landmarks))

        landmarkIndices = landmarkMap[args.landmarks]

        # align image
        outRgb = self.dlib_aligner.align(args.size, image,
                             landmarkIndices=landmarkIndices,
                             skipMulti=args.skipMulti)
        if outRgb is None:
            print("--- Unable to align.")

        return outRgb