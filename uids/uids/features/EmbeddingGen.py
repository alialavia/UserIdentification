#!/usr/bin/env python2
import os
import time
import numpy as np
import openface
import openface.helper
from openface.data import iterImgs
from uids.utils.Logger import Logger as log

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')


class EmbeddingGen:
    """
    INPUT FORMATS:
    - aligned (face alignment)
    - RGB image, size 96 squared
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
        log.info('cnn', "EmbeddingGen: loading models...")
        # load neural net
        self.neural_net = openface.TorchNeuralNet(self.networkModel, imgDim=self.size, cuda=self.cuda)
        # load dlib model
        self.dlib_aligner = openface.AlignDlib(dlibModelDir + "/" + self.dlibFacePredictor)
        log.info('cnn', "EmbeddingGen: model loading took {} seconds".format("%.3f" % (time.time() - start)))

    #  ----------- EMBEDDING GENERATION
    def get_embeddings(self, rgb_images, align=True):
        """
        Calculate deep face embeddings for input images
        :param rgb_images: RGB (!) images
        :param align:
        :return: np.array embedding vectors
        """

        images_normalized = []
        embeddings = []
        start = time.time()

        # normalize images
        if align is True:
            if len(rgb_images) > 0:
                for imgObject in rgb_images:
                    # align face - ignore images with multiple bounding boxes
                    aligned = self.align_face(imgObject, self.landmarks, self.size)
                    if aligned is not None:
                        images_normalized.append(aligned)

            # print status
            if self.verbose is True:
                if len(images_normalized) > 0:
                    log.debug('cnn', "Alignment took {} seconds - {}/{} images suitable".format(time.time() - start, len(images_normalized), len(rgb_images)))
                else:
                    log.warning("No suitable images (no faces detected)")
                    return np.array(embeddings)
        else:
            images_normalized = rgb_images

        # generate embeddings
        start = time.time()
        for img in images_normalized:
            rep = self.neural_net.forward(img)
            embeddings.append(rep)

        # if self.verbose:
        #     print("--- Neural network forward pass took {} seconds.".format(time.time() - start))

        return np.array(embeddings)

    def get_embedding(self, rgb_img, align=True):
        # align image
        if align:
            normalized = self.align_face(rgb_img, self.landmarks, self.size)
            if normalized is None:
                return None
        else:
            normalized = rgb_img

        # generate embedding
        rep = self.neural_net.forward(normalized)
        return rep

    def align_face(self, rgb_image, landmark, output_size, skip_multi=False):

        landmarkMap = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if landmark not in landmarkMap:
            raise Exception("Landmarks unrecognized: {}".format(landmark))

        landmarkIndices = landmarkMap[landmark]

        # align image (needs RGB image, not BGR)
        outRgb = self.dlib_aligner.align(output_size, rgb_image,
                                         landmarkIndices=landmarkIndices,
                                         skipMulti=skip_multi)

        # out Rgb might be none
        return outRgb
