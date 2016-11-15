#!/usr/bin/env python2

import argparse
import cv2
import numpy as np
import os
import random
import shutil
import time

import openface
import openface.helper
from openface.data import iterImgs

REQUEST_LOOKUP = {
    1: 'identification',            # request user id
    2: 'receive_training_images',   # receive training images
    3: 'embedding_calculation',     # direct embedding calculation
    4: 'classifier_training',       # initialize classifier training
    5: 'image_normalization'        # face normalization
}

# tcp networking
from lib.TCPServer import TCPServerBlocking
# classifier
from lib.OfflineUserClassifier import OfflineUserClassifier

class TCPTestServer(TCPServerBlocking):

    classifier = None

    def __init__(self, host, port):
        TCPServerBlocking.__init__(self, host, port)

        # initialize classifier
        self.classifier = OfflineUserClassifier()

    def handle_request(self, conn, addr):
        """general request handler"""
        request_id = self.receive_char(conn)
        if(request_id in REQUEST_LOOKUP):
            request = REQUEST_LOOKUP[request_id]
            print '=== Request: ' + request
            if request_id == 1:     # identification
                self.handle_identification(conn)
            elif request_id == 2:   # send training images
                self.handle_embedding_collection(conn)
            elif request_id == 3:   # embedding calculation
                self.handle_embedding_calculation(conn)
            elif request_id == 4:   # classifier training
                self.handle_classifier_training(conn)
            elif request_id == 5:   # image normalization
                self.handle_image_normalization(conn)
            else:
                print '=== Invalid request identifier, shutting down server...'
                self.SERVER_STATUS = -1  # shutdown server

        # communication finished - close connection
        conn.close()

    #  ----------- REQUEST HANDLERS

    def handle_identification(self, conn):
        print "--- Identification"

        # receive image size
        img_size = self.receive_int(conn)
        # receive image
        user_face = self.receive_rgb_image(conn, img_size, img_size)
        # identify
        user_id, confidence = self.classifier.identify_user(user_face)
        # send back user id
        self.send_uint(conn, user_id)
        # send back confidence
        self.send_float(conn, confidence)

    def handle_classifier_training(self, conn):
        print "--- Classifier Training"
        self.classifier.trigger_training()

    def handle_embedding_collection(self, conn):

        # receive user id
        user_id = self.receive_char(conn)

        # receive image size
        img_size = self.receive_int(conn)

        # receive batch size
        nr_images = self.receive_char(conn)

        print "--- Image batch received: size: " + str(nr_images) + " | image size: " + str(img_size)

        # receive image batch
        images = []
        for x in range(0, nr_images):
            # receive image
            new_img = self.receive_rgb_image(conn, img_size, img_size)
            images.append(new_img)

        # forward to classifier
        self.classifier.collect_embeddings(images, user_id)

    def handle_image_normalization(self, conn):
        # receive image size
        img_size = self.receive_int(conn)
        img = self.receive_rgb_image(conn, img_size, img_size)
        # normalize
        normalized = self.classifier.align_face(img, 'outerEyesAndNose', 96)
        if normalized is not None:
            # send image back
            self.send_rgb_image(conn, normalized)
        else:
            print "Image could not be aligned"

    def handle_embedding_calculation(self, conn):
        print "--- Direct Embeddings Calculation"

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


# ================================= #
#              Main

if __name__ == '__main__':

    server = TCPTestServer('', 8080)
    server.start_server()
