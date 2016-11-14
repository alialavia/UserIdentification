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
    1: 'identification',        # request user id
    2: 'training',              # direct classifier training
    3: 'embedding_calculation', # direct embedding calculation
    4: 'classifier_training',   # initialize classifier training
    5: 'image_normalization'    # face normalization
}

# tcp networking
from lib.TCPServer import TCPServer

class TCPTestServer(TCPServer):

    def __init__(self, host, port):
        TCPServer.__init__(self, host, port)

    def handle_request(self, conn, addr):
        """general request handler"""
        request_id = self.receive_char(conn)

        if(request_id in REQUEST_LOOKUP):
            request = REQUEST_LOOKUP[request_id]
            print '=== Request: ' + request
            if request_id == 1:
                self.handle_identification(conn)
            elif request_id == 2:
                self.handle_training(conn)
            elif request_id == 3:
                self.handle_embedding_calculation(conn)
            elif request_id == 4:
                self.handle_classifier_training(conn)
            elif request_id == 5:
                self.handle_image_normalization(conn)
            else:
                print '=== Invalid request identifier, shutting down server...'
                self.SERVER_STATUS = -1  # shutdown server

        # communication finished - close connection
        # conn.close()

    #  ----------- REQUEST HANDLERS

    def handle_classifier_training(self, conn):
        print "--- Classifier Training"

    def handle_training(self, conn):
        print "--- Training"

    def handle_identification(self, conn):
        print "--- Identification"

    def handle_image_normalization(self, conn):
        # receive image size
        img_size = self.receive_integer(conn)

        # receive batch size
        nr_images = self.receive_char(conn)

        print "--- Batch size: " + str(nr_images) + " | image size: " + str(img_size)

        """Normalize face patch and send back to client"""
        args = Arguments()
        new_img = self.receive_rgb_image(conn, 96, 96)
        # normalize
        aligned = self.align_face(args, new_img)
        if aligned is not None:
            # send image back
            self.send_rgb_image(conn, aligned)
        else:
            print "Image could not be aligned"

    def handle_embedding_calculation(self, conn):

        args = Arguments()

        # receive user id
        user_id = self.receive_char(conn)

        # receive image size
        img_size = self.receive_integer(conn)

        # receive batch size
        nr_images = self.receive_char(conn)

        print "--- Image batch received: size: " + str(nr_images) + " | image size: " + str(img_size)

        # receive image batch
        images = []
        for x in range(0, nr_images):
            # receive image
            new_img = self.receive_rgb_image(conn, img_size, img_size)
            images.append(new_img)

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

        # generate embedding
        net = openface.TorchNeuralNet(args.networkModel, imgDim=img_size,
                                      cuda=args.cuda)
        # representations
        reps = []
        for img in images_normalized:
            start = time.time()
            rep = net.forward(img)
            print("--- = Neural network forward pass took {} seconds.".format(
                time.time() - start))
            reps.append(rep)

        # print
        # for rep in reps:
        #     print "= rep: " + str(rep)

        # save
        if user_id in self.user_embeddings:
            # append
            self.user_embeddings[user_id].append(reps)
        else:
            self.user_embeddings[user_id] = reps

        # display current representations
        self.print_embedding_status()

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

if __name__=='__main__':

    server = TCPTestServer('', 8080)
    server.start_server()
