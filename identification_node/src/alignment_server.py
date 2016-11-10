#!/usr/bin/env python2

import argparse
import cv2
import numpy as np
import os
import random
import shutil

import openface
import openface.helper
from openface.data import iterImgs

REQUEST_LOOKUP = {
    1: 'training',
    2: 'identification'
}

# tcp networking
from lib.TCPServer import TCPServer

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')

class TCPTestServer(TCPServer):

    def __init__(self, host, port):
        TCPServer.__init__(self, host, port)


    def handle_request(self, conn, addr):
        """general request handler"""
        request_id = self.receive_char(conn)

        if(request_id in REQUEST_LOOKUP):
            request = REQUEST_LOOKUP[request_id]

            if request_id == 1:
                print '--- '+str(request_id)+': Normalizing image...'
                self.handle_training(conn)
            elif request_id == 2:
                print '--- '+str(request_id)+': Identification...'
            else:
                print '--- Invalid request identifier, shutting down server...'
                self.SERVER_STATUS = -1  # shutdown server

        # communication finished - close connection
        conn.close()

    #  ----------- REQUEST HANDLERS

    def handle_training(self, conn):
        args = {}
        args.dlibFacePredictor = "shape_predictor_68_face_landmarks.dat"  # Path to dlib's face predictor.
        args.landmarks = "outerEyesAndNose"
        choices = ['outerEyesAndNose',
                   'innerEyesAndBottomLip',
                   'eyes_1']
        args.size = 96  # Default image size.
        args.skipMulti = True  # "Skip images with more than one face."
        args.verbose = True

        # image batch
        images = []

        # receive image batch
        new_img = self.receive_rgb_image(conn, 96, 96)
        images.append(new_img)

        # normalize images
        if len(images) > 0:
            # random.shuffle(images)
            # do alignment
            for imgObject in images:
                aligned = self.align_face(args, imgObject)
                # send image back
                self.send_rgb_image(conn, aligned)

    def handle_image(self, conn):
        """receive image, draw and send back"""
        img = self.receive_rgb_image(conn, 100, 100)
        height, width, channels = img.shape
        # display image
        cv2.imshow('Server image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # draw circle in the center
        cv2.circle(img, (width/2, height/2), height/4, (0, 0, 255), -1)
        # send image back
        self.send_rgb_image(conn, img)

    def align_face(self, args, image):

        landmarkMap = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if args.landmarks not in landmarkMap:
            raise Exception("Landmarks unrecognized: {}".format(args.landmarks))

        landmarkIndices = landmarkMap[args.landmarks]

        # dlib aligner
        align = openface.AlignDlib(args.dlibFacePredictor)
        outRgb = align.align(args.size, image,
                             landmarkIndices=landmarkIndices,
                             skipMulti=args.skipMulti)
        if outRgb is None:
            print("  + Unable to align.")

        return outRgb


# ================================= #
#              Main

if __name__=='__main__':

    server = TCPTestServer('', 8080)
    server.start_server()
