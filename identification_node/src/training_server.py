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

    # 4 byte: 0 .. 4294967296
    def sendUnsignedInteger(self, the_socket, short):
        # 4-byte length
        # convert to network byte order
        msg = struct.pack('!h', int)
        the_socket.send(msg)

    #  --------------------------------------- DEPRECATED

    def sendMessageWithLength(sock, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('!I', len(msg)) + msg
        sock.sendall(msg)

    def send_integer(self, client_socket):
        # must be in range 0-255
        number = 5
        client_socket.send(chr(number))

# ================================= #
#              Main

if __name__=='__main__':

    server = TCPServer('', 555)
    server.start_server()
