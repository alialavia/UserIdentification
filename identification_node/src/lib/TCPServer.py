#!/usr/bin/python
import numpy
import struct
import sys
import socket
from abc import abstractmethod


class TCPServer:
    """
        TCP Server Interface

        USAGE:
        server = TCPServer('', 555)
        server.start_server()
    """

    HOST = ''     # Symbolic name meaning all available interfaces
    PORT = '8080'  # Arbitrary non-privileged port
    SERVER_SOCKET = -1
    SERVER_STATUS = 0
    STATUS_CLEAN = {
        -1: 'shutdown',
        0: 'starting',
        1: 'running'
    },

    def __init__(self, host, port):
        self.HOST = host
        self.PORT = port

    def start_server(self):

        self.SERVER_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # create socket
        try:
            self.SERVER_SOCKET.bind((self.HOST, self.PORT))
        except socket.error, msg:
            print '--- Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
            sys.exit()

        # begin listening to connections
        self.SERVER_SOCKET.listen(5)
        self.SERVER_STATUS = 1
        print '--- Server started on port ', self.PORT

        # server loop
        self.server_loop()

    @abstractmethod
    def handle_request(self, conn, addr):
        """Handles the socket request in each loop"""
        raise NotImplementedError( "The basic request handler must be implemented first." )

    @abstractmethod
    def server_loop(self):
        """The main processing loop - implement in server type (blocking/parallel)"""
        raise NotImplementedError( "The basic processling loop must be implemented in the server type class." )

    #  ----------- MESSAGE HANDLERS

    def receive_message(self, the_socket, datasize):
        """Basic message receiver for known datasize"""
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

    #  ----------- IMAGE HANDLERS

    def receive_rgb_image(self, client_socket, width, height):
        """receive 8 bit rgb image"""
        # 3 channels, 8 bit = 1 byte
        string_data = self.receive_message(client_socket, width * height * 3)
        data = numpy.fromstring(string_data, dtype='uint8')
        reshaped = data.reshape((width, height, 3))
        return reshaped

    def send_rgb_image(self, client_socket, img):
        """send 8 bit rgb image"""
        data = numpy.array(img)
        stringData = data.tostring()
        # send image size
        # client_socket.send(str(len(stringData)).ljust(16))
        client_socket.send(stringData)

        #  ----------- BINARY DATA HANDLERS

    def receive_char(self, client_socket):
        """1 byte - unsigned: 0 .. 255"""
        # read 1 byte = char = 8 bit (2^8), BYTE datatype: minimum value of -127 and a maximum value of 127
        raw_msg = self.receive_message(client_socket, 1)
        if not raw_msg:
            return None
        # 8-bit string to integer
        request_id = ord(raw_msg)
        # print 'Client request ID: ' + str(request_id)
        return request_id

    def receive_integer(self, client_socket):
        """read 4 bytes"""
        raw_msglen = self.receive_message(client_socket, 4)
        if not raw_msglen:
            return None
        # network byte order
        msglen = struct.unpack('!i', raw_msglen)[0]  # convert to host byte order
        return msglen

    def send_unsigned_integer(self, the_socket, int):
        """4 byte, 32bit: 0 .. 4294967296"""
        msg = struct.pack('!I', int)  # convert to network byte order
        the_socket.send(msg)

    def send_unsigned_short(self, the_socket, short):
        """"2 byte, 16bit: 0 .. 65535"""
        msg = struct.pack('!H', short)  # convert to network byte order
        the_socket.send(msg)


class TCPServerBlocking(TCPServer):
    """Blocking TCP Server - 1 socket connected at a time"""
    def __init__(self, host, port):
        TCPServer.__init__(self, host, port)

    def server_loop(self):
        # server loop
        while True:
            # accept new connection - blocking call, wait for new socket to connect to
            conn, addr = self.SERVER_SOCKET.accept()
            print '--- Connected with ' + addr[0] + ':' + str(addr[1])

            # handle request
            self.handle_request(conn, addr)

            # check status - eventually shutdown server
            if self.SERVER_STATUS == -1:
                conn.close()    # close connection
                break

            # close connection - allow new socket connections
            conn.close()
