#!/usr/bin/env python2
import os
import time
import openface
import openface.helper
from openface.data import iterImgs

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')


class EmbeddingGen:
    """
    TODO:
    - implement embedding confidence (based on normalization measure)
    """

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
        print "--- EmbeddingGen: loading models..."
        # load neural net
        self.neural_net = openface.TorchNeuralNet(self.networkModel, imgDim=self.size, cuda=self.cuda)
        # load dlib model
        self.dlib_aligner = openface.AlignDlib(dlibModelDir + "/" + self.dlibFacePredictor)
        print("--- EmbeddingGen: model loading took {} seconds".format(time.time() - start))

    #  ----------- EMBEDDING GENERATION
    def get_embeddings(self, input_images, align=True):

        images_normalized = []
        embeddings = []
        start = time.time()

        # normalize images
        if align is True:
            if len(input_images) > 0:
                for imgObject in input_images:
                    # align face - ignore images with multiple bounding boxes
                    aligned = self.__align_face(imgObject, self.landmarks, self.size)
                    if aligned is not None:
                        images_normalized.append(aligned)

            # print status
            if self.verbose is True:
                if len(images_normalized) > 0:
                    print("--- Alignment took {} seconds - {}/{} images suitable".format(time.time() - start, len(images_normalized), len(input_images)))
                else:
                    print "--- No suitable images (no faces detected)"
                    return embeddings
        else:
            images_normalized = input_images

        # generate embeddings
        start = time.time()
        for img in images_normalized:
            rep = self.neural_net.forward(img)
            embeddings.append(rep)

        if self.verbose:
            print("--- Neural network forward pass took {} seconds.".format(time.time() - start))

        return embeddings

    def get_embedding(self, user_img):
        # align image
        normalized = self.__align_face(user_img, self.landmarks, self.size)
        if normalized is None:
            return None

        # generate embedding
        rep = self.neural_net.forward(normalized)
        return rep

    #  ----------- TOOLS
    def __align_face(self, image, landmark, output_size, skip_multi=False):

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

        # out Rgb might be none
        return outRgb
