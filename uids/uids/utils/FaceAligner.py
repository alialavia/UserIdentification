import os
import time
import numpy as np
from uids.utils.Logger import Logger as log


# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model


class FaceAligner:

    # settings
    dlibFacePredictor = "shape_predictor_68_face_landmarks.dat"
    landmarks = "outerEyesAndNose"
    __aligner = None

    def __init__(self):
        # load dlib model
        # TODO: implement without openface
        self.__aligner = openface.AlignDlib(dlibModelDir + "/" + self.dlibFacePredictor)

    def align_face(self, image, output_size, skip_multi=False):

        # landmarkMap = {
        #     'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        #     'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        # }
        # if landmark not in landmarkMap:
        #     raise Exception("Landmarks unrecognized: {}".format(landmark))
        # landmarkIndices = landmarkMap[landmark]

        INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
        OUTER_EYES_AND_NOSE = [36, 45, 33]

        # align image
        out = self.__aligner.align(output_size, image,
                                         landmarkIndices=OUTER_EYES_AND_NOSE,
                                         skipMulti=skip_multi)

        # might be none
        return out


