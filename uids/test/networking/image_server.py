#!/usr/bin/python
import socket
import cv2
import numpy
import time
import sys
import struct

HOST = ''  # Symbolic name meaning all available interfaces
PORT = 80  # Arbitrary non-privileged port

# ================================= #
#              Definitions

def recv_basic(the_socket):
    total_data=[]
    while True:
        data = the_socket.recv(30000)
        if not data: break
        total_data.append(data)
    return ''.join(total_data)


def start_server():
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    # create socket
    try:
        sock.bind((HOST, PORT))
    except socket.error, msg:
        print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
        sys.exit()

    # begin listening to connections
    sock.listen(5)
    print 'Server started on port ', PORT

    # server loop
    while True:

        # new socket connectd
        newsock, addr = sock.accept()
        print 'Connected with ' + addr[0] + ':' + str(addr[1])

        # processing
        stringData = recv_basic(newsock)
        data = numpy.fromstring(stringData, dtype='uint8')
        decimg = data.reshape((100, 100, 3))

        # display image
        cv2.imshow('SERVER', decimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ================================= #
#              Main

if __name__=='__main__':
    start_server()
