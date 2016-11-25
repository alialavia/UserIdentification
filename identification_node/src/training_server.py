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
        request_id = self.receive_uchar(conn)
        if(request_id in REQUEST_LOOKUP):
            request = REQUEST_LOOKUP[request_id]
            print '=== Request: ' + request
            if request_id == 1:     # identification
                self.handle_identification(conn)
            elif request_id == 2:   # send training images
                self.handle_embedding_collection_by_nice_name(conn)
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
        img_size = self.receive_uint(conn)
        # receive image
        user_face = self.receive_rgb_image(conn, img_size, img_size)
        # identify
        user_id, nice_name, confidence = self.classifier.identify_user(user_face)

        if (user_id is None):
            print "--- Identification Error"
            # send back error response
            self.send_int(conn, 99)
            return

        print "--- User ID: " + str(user_id) + " | confidence: " + str(confidence)

        # send back response type
        self.send_int(conn, 1)

        # send back user id
        self.send_int(conn, int(user_id))

        # send back nice name
        self.send_string(conn, nice_name)

        # send back confidence
        self.send_float(conn, float(confidence))

    def handle_classifier_training(self, conn):
        print "--- Classifier Training"
        self.classifier.trigger_training()

    def handle_embedding_collection_by_nice_name(self, conn):

        # receive user name size
        user_name_bytes = self.receive_int(conn)

        # receive user nice name
        user_nice_name = self.receive_message(conn, user_name_bytes)

        # receive image dimension
        img_dim = self.receive_int(conn)

        # receive batch size
        nr_images = self.receive_char(conn)

        print "--- Image batch received: size: " + str(nr_images) + " | image dimension: " + str(img_dim)

        # receive image batch
        images = []
        for x in range(0, nr_images):
            # receive image
            new_img = self.receive_rgb_image(conn, img_dim, img_dim)
            images.append(new_img)

        # get user id from nice name
        # ASSUME USER NAME IS UNIQUE!
        user_id = self.classifier.get_user_id_from_name(user_nice_name)

        if user_id == 0:
            # generate new user id
            print "create new user"
            user_id = self.classifier.create_new_user(user_nice_name)
            # save nice name

        # forward to classifier
        self.classifier.collect_embeddings_for_specific_id(images, user_id)

    # def handle_embedding_collection_by_id(self, conn):
    #
    #     # receive user id size
    #     user_id_bytes = self.receive_int(conn)
    #
    #     # receive user id
    #     user_id = self.receive_message(conn, user_id_bytes)
    #
    #     # receive image dimension
    #     img_dim = self.receive_int(conn)
    #
    #     # receive batch size
    #     nr_images = self.receive_char(conn)
    #
    #     print "--- Image batch received: size: " + str(nr_images) + " | image dimension: " + str(img_dim)
    #
    #     # receive image batch
    #     images = []
    #     for x in range(0, nr_images):
    #         # receive image
    #         new_img = self.receive_rgb_image(conn, img_dim, img_dim)
    #         images.append(new_img)
    #
    #     # forward to classifier
    #     self.classifier.collect_embeddings(images, user_id)

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help="Server port.", default=8080)

    args = parser.parse_args()

    server = TCPTestServer('', args.port)
    server.start_server()
