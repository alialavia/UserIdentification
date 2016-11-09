#!/usr/bin/python
import socket
import cv2
import numpy
import time
import sys
import struct
import sys
import socket
import os
import errno
from time import sleep


import argparse
import cv2
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


REQUEST_LOOKUP = {
    1: 'identification',
    2: 'offline_training',
    3: 'online_training'
}

class TCPServer:
    HOST = ''     # Symbolic name meaning all available interfaces
    PORT = '555'  # Arbitrary non-privileged port
    SERVER_SOCKET = -1

    def __init__(self, host, port):
        self.HOST = host
        self.PORT = port

    def start_server(self):

        self.SERVER_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # non-blocking asynchronous communication
        # socket.setblocking(0)

        # create socket
        try:
            self.SERVER_SOCKET.bind((self.HOST, self.PORT))
        except socket.error, msg:
            print '--- Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
            sys.exit()

        # begin listening to connections
        self.SERVER_SOCKET.listen(5)
        print '--- Server started on port ', self.PORT

        # server loop
        while True:
            # new socket connected
            conn, addr = self.SERVER_SOCKET.accept()
            print '--- Connected with ' + addr[0] + ':' + str(addr[1])

            # wait to receive request id
            request_id = self.receiveChar(conn)

            print '--- Request ID: ' + str(request_id)

            message_length = self.receiveInteger(conn)

            print '--- Message of length ' + str(message_length) + " received."


            print '--- Sending ID to client'
            self.sendUnsignedInteger(conn, 4294967295)


            if(request_id in REQUEST_LOOKUP):
                request = REQUEST_LOOKUP[request_id]
                if request == 'offline_training':
                    print '--- Do offline training'
                elif request == 'online_training':
                    print '--- Do online training'
                elif request == 'identification':
                    print '--- Do identification'
                else:
                    print '--- Request Handling not yet implemented for: '.request
            else:
                print '--- Invalid request identifier, shutting down server...'
                break

            # processing
            # stringData = self.recv_basic(conn, 30000)
            #
            # print 'Received ' + str(sys.getsizeof(stringData)) + ' bytes'
            #
            # print '--- Image received...'
            # data = numpy.fromstring(stringData, dtype='uint8')
            # decimg = data.reshape((100, 100, 3))
            #
            # # display image
            # cv2.imshow('SERVER', decimg)
            # cv2.waitKey(2000)
            # cv2.destroyAllWindows()
            #
            # # short int - network byte order
            # short = struct.pack('!h', 3)
            # conn.send(short)
            # # conn.send(b'hey theere')
            # print '--- Reply sent...'

            # communication finished - close connection
            conn.close()

    """Message Receiving"""

    def receiveMessage(self, the_socket, datasize):
        buffer = ''
        try:
            while len(buffer) < datasize:
                packet = the_socket.recv(datasize - len(buffer))
                # read-in finished too early - return None
                if not packet:
                    return None
                # append to buffer
                buffer += packet
                # print 'Total ' + str(sys.getsizeof(buffer)) + ' bytes'
        except socket.error, (errorCode, message):
            # error 10035 is no data available, it is non-fatal
            if errorCode != 10035:
                print 'socket.error - (' + str(errorCode) + ') ' + message
        return buffer

    #  --------------------------------------- IMAGE HANDLERS

    def receiveRGB8Image(self, client_socket, width, height):
        # 3 channels, 8 bit = 1 byte
        string_data = self.receiveMessage(client_socket, width * height * 3)
        data = numpy.fromstring(string_data, dtype='uint8')
        reshaped = data.reshape((width, height, 3))
        return reshaped

    def receiveRGB16Image(self, client_socket, width, height):
        # 3 channels, 16 bit = 2 byte
        string_data = self.receiveMessage(client_socket, 2 * width * height * 3)
        data = numpy.fromstring(string_data, dtype='uint16')
        reshaped = data.reshape((width, height, 3))
        return reshaped

    #  --------------------------------------- BINARY DATA HANDLERS

    # 1 byte - unsigned: 0 .. 255
    def receiveChar(self, client_socket):
        # read 1 byte = char = 8 bit (2^8), BYTE datatype: minimum value of -127 and a maximum value of 127
        raw_msg = self.receiveMessage(client_socket, 1)
        if not raw_msg:
            return None
        # 8-bit string to integer
        request_id = ord(raw_msg)
        # print 'Client request ID: ' + str(request_id)
        return request_id

    # 4 byte
    def receiveInteger(self, client_socket):
        # read 4 bytes
        raw_msglen = self.receiveMessage(client_socket, 4)
        if not raw_msglen:
            return None
        # network byte order
        msglen = struct.unpack('!i', raw_msglen)[0]
        return msglen

    # 4 byte: 0 .. 4294967296
    def sendUnsignedInteger(self, the_socket, int):
        # 4-byte length
        # convert to network byte order
        msg = struct.pack('!I', int)
        the_socket.send(msg)

    #  --------------------------------------- DEPRECATED

# ================================= #
#              CLASSIFIER TRAINING

def train(args):

    # load image labels
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.


    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    if args.classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
    elif args.classifier == 'GridSearchSvm':
        print("""
        Warning: In our experiences, using a grid search over SVM hyper-parameters only
        gives marginally better performance than a linear SVM with C=1 and
        is not worth the extra computations of performing a grid search.
        """)
        param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
        ]
        clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
    elif args.classifier == 'GMM':  # Doesn't work best
        clf = GMM(n_components=nClasses)

    # ref:
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
    elif args.classifier == 'RadialSvm':  # Radial Basis Function kernel
        # works better with C = 1 and gamma = 2
        clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
    elif args.classifier == 'DecisionTree':  # Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)
    elif args.classifier == 'GaussianNB':
        clf = GaussianNB()

    # ref: https://jessesw.com/Deep-Learning/
    elif args.classifier == 'DBN':
        from nolearn.dbn import DBN
        clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                  learn_rates=0.3,
                  # Smaller steps mean a possibly more accurate result, but the
                  # training will take longer
                  learn_rate_decays=0.9,
                  # a factor the initial learning rate will be multiplied by
                  # after each iteration of the training
                  epochs=300,  # no of iternation
                  # dropouts = 0.25, # Express the percentage of nodes that
                  # will be randomly dropped as a decimal.
                  verbose=1)

    if args.ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                        ('clf', clf_final)])

    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(args.workDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)




# ================================= #
#              Main

if __name__=='__main__':

    server = TCPServer('', 555)
    server.start_server()

    # set arguments
    args = {}
    args.classifier = 'LinearSvm'
    args.dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
    args.networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7')
    args.imgDim = 96
    args.cuda = None

    # specify face predictor
    align = openface.AlignDlib(args.dlibFacePredictor)
    # specify neural network model
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)

    args.workDir =
    

    # train the classifier
    train(args)

