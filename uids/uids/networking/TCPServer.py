#!/usr/bin/python
import numpy
import struct
import sys
import socket
import time
from abc import abstractmethod


class TCPServer:
    """
        TCP Server Interface

        USAGE:
        server = TCPServer('', 555)
        server.start_server()

        DEFINITIONS:
        request/response type id:
            - request: uchar (0-255)
            - response: uint
        images:
            - receival/send order: nr_images (short), image_width (short), image_height (short), images
            - receival/send order (changing size): nr_images (short), Loop(image_width (short), image_height (short), image)

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

    #  ----------- DYNAMIC MESSAGE HANDLERS

    def send_string(self, target_socket, msg):

        # send message length
        self.send_int(target_socket, len(msg))

        # send string
        totalsent = 0
        while totalsent < len(msg):
            sent = target_socket.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    #  ----------- HELPER METHODS

    def receive_message(self, the_socket, datasize, timeout=2):
        """Basic message receiver for known datasize"""
        buffer = ''
        begin = time.time()
        refresh_rate = 0.01

        try:
            while len(buffer) < datasize:
                # if you got some data, then break after timeout
                if buffer and time.time() - begin > timeout:
                    raise ValueError('receive_message timeout. Only partial data received.')
                # if you got no data at all, wait a little longer, twice the timeout
                elif time.time() - begin > timeout * 2:
                    raise ValueError('receive_message timeout. No data received.')

                packet = the_socket.recv(datasize - len(buffer))

                if packet:
                    # append to buffer
                    buffer += packet
                    # print 'Total ' + str(sys.getsizeof(buffer)) + ' bytes'
                    begin = time.time()
                if not packet:
                    # wait
                    time.sleep(refresh_rate)

        except socket.error, (errorCode, message):
            # error 10035 is no data available, it is non-fatal
            if errorCode != 10035:
                print 'socket.error - (' + str(errorCode) + ') ' + message
        return buffer

    #  ----------- IMAGE HANDLERS

    def receive_rgb_image(self, client_socket, width=None, height=None):
        """receive 8 bit rgb image"""

        if height is None:
            if width is None:
                # variable image width
                width = self.receive_short(client_socket)
                height = self.receive_short(client_socket)
            else:
                # squared image
                height = width

        # 3 channels, 8 bit = 1 byte
        string_data = self.receive_message(client_socket, width * height * 3)
        data = numpy.fromstring(string_data, dtype='uint8')
        reshaped = data.reshape((width, height, 3))
        return reshaped

    def receive_rgb_image_squared(self, client_socket, size=None):
        if size is None:
            size = self.receive_short(client_socket)
        return self.receive_rgb_image(client_socket, size)

    def send_rgb_image(self, client_socket, img):
        """send 8 bit rgb image"""
        data = numpy.array(img)
        stringData = data.tostring()
        # send image size
        cols, rows = img.shape()
        self.send_short(client_socket, cols)
        self.send_short(client_socket, rows)
        client_socket.send(stringData)

    def send_rgb_image_squared(self, client_socket, img):
        data = numpy.array(img)
        stringData = data.tostring()
        # send image size
        cols, rows, channels = img.shape
        if cols != rows:
            raise ValueError('Trying to send image that is not quadratic!')
        self.send_short(client_socket, cols)
        client_socket.send(stringData)

    def send_image_batch_squared_same_size(self, client_socket, images):

        if len(images) == 0:
            return

        # send image number
        self.send_short(client_socket, len(images))

        # send image size
        cols, rows = images[0].shape()
        if cols != rows:
            raise ValueError('Trying to send image that is not quadratic!')

        # send image size
        self.send_short(client_socket, cols)

        # send data
        string_data = ""
        for img in images:
            data = numpy.array(img)
            string_data += data.tostring()
        client_socket.send(string_data)

        #  ----------- BINARY DATA HANDLERS

    def receive_image_batch_squared_same_size(self, client_socket):

        # receive batch size
        nr_images = self.receive_short(client_socket)

        # receive image dimension
        img_dim = self.receive_short(client_socket)

        # receive image batch
        images = []
        for x in range(0, nr_images):
            # receive image
            new_img = self.receive_rgb_image(client_socket, img_dim, img_dim)
            images.append(new_img)

        return images

    def receive_image_batch_squared(self, client_socket):

        # receive batch size
        nr_images = self.receive_short(client_socket)

        # receive image batch
        images = []
        for x in range(0, nr_images):
            # receive image dimension
            img_dim = self.receive_short(client_socket)
            # receive image
            new_img = self.receive_rgb_image(client_socket, img_dim, img_dim)
            images.append(new_img)

        return images

    def receive_image_batch_same_size(self, client_socket):

        # receive batch size
        nr_images = self.receive_short(client_socket)

        # image dimensions
        w = self.receive_short(client_socket)
        h = self.receive_short(client_socket)

        # receive image batch
        images = []
        for x in range(0, nr_images):
            # receive image
            new_img = self.receive_rgb_image(client_socket, w, h)
            images.append(new_img)

        return images

    def receive_image_batch(self, client_socket):

        # receive batch size
        nr_images = self.receive_short(client_socket)

        # receive image batch
        images = []
        for x in range(0, nr_images):
            # image dimensions
            w = self.receive_short(client_socket)
            h = self.receive_short(client_socket)
            # receive image
            new_img = self.receive_rgb_image(client_socket, w, h)
            images.append(new_img)

        return images

    #  ----------- RECEIVE PRIMITIVES
    def receive_string(self, client_socket):
        # get message size
        msg_size = self.receive_uint(client_socket)
        msg = self.receive_message(client_socket, msg_size)
        return msg

    def receive_char(self, client_socket):
        """1 byte - unsigned: 0 .. 255"""
        # read 1 byte = char = 8 bit (2^8), BYTE datatype: minimum value of -127 and a maximum value of 127
        raw_msg = self.receive_message(client_socket, 1)
        if not raw_msg:
            return None
        # 8-bit string to integer
        numerical = ord(raw_msg)
        return numerical

    def receive_uchar(self, client_socket):
        return self.receive_char(client_socket)

    def receive_short(self, client_socket):
        """read 2 bytes"""
        raw_msglen = self.receive_message(client_socket, 2)
        if not raw_msglen:
            return None
        # network byte order
        msglen = struct.unpack('!h', raw_msglen)[0]  # convert to host byte order
        return msglen

    def receive_ushort(self, client_socket):
        """read 2 bytes"""
        raw_msglen = self.receive_message(client_socket, 2)
        if not raw_msglen:
            return None
        # network byte order
        msglen = struct.unpack('!H', raw_msglen)[0]  # convert to host byte order
        return msglen

    def receive_int(self, client_socket):
        """read 4 bytes"""
        raw_msglen = self.receive_message(client_socket, 4)
        if not raw_msglen:
            return None
        # network byte order
        msglen = struct.unpack('!i', raw_msglen)[0]  # convert to host byte order
        return msglen

    def receive_uint(self, client_socket):
        """read 4 bytes"""
        raw_msglen = self.receive_message(client_socket, 4)
        if not raw_msglen:
            return None
        # network byte order
        msglen = struct.unpack('!I', raw_msglen)[0]  # convert to host byte order
        return msglen

    def receive_bool(self, client_socket):
        """bool saced as char, 1 byte"""
        char = client_socket.recv(1)
        return bool(ord(char))

    def receive_float(self, client_socket):
        """read 4 bytes - python float equals c double. When receiving C floats you loose precision"""
        raw_msglen = self.receive_message(client_socket, 4)
        if not raw_msglen:
            return None
        # network byte order
        msglen = struct.unpack('!f', raw_msglen)[0]  # convert to host byte order
        return msglen

    def receive_double(self, client_socket):
        """read 8 bytes - python float equals c double."""
        raw_msglen = self.receive_message(client_socket, 8)
        if not raw_msglen:
            return None
        # network byte order
        msglen = struct.unpack('!d', raw_msglen)[0]  # convert to host byte order
        return msglen

    #  ----------- SEND PRIMITIVES

    def send_char(self, target_socket, char):
        msg = struct.pack('c', chr(char))
        target_socket.send(msg)

    def send_uchar(self, target_socket, uchar):
        target_socket.send(chr(uchar))

    def send_short(self, target_socket, short):
        """"2 byte, 16bit: -32,768 .. 32,767"""
        msg = struct.pack('!h', short)  # convert to network byte order
        target_socket.send(msg)

    def send_ushort(self, target_socket, short):
        """"2 byte, 16bit: 0 .. 65535"""
        msg = struct.pack('!H', short)  # convert to network byte order
        target_socket.send(msg)

    def send_int(self, target_socket, int):
        """"4 byte, 32bit: -2,147,483,648 .. 2,147,483,647"""
        msg = struct.pack('!i', int)  # convert to network byte order
        target_socket.send(msg)

    def send_uint(self, target_socket, uint):
        """"4 byte, 32bit: 0 .. 4,294,967,295"""
        msg = struct.pack('!I', uint)  # convert to network byte order
        target_socket.send(msg)

    def send_bool(self, target_socket, bool):
        """1 byte, 8 bit"""
        msg = struct.pack('?', bool)
        target_socket.send(msg)

    def send_float(self, target_socket, val):
        """4 byte, 32bit: 10^-38 .. 10^38"""
        msg = struct.pack('!f', val)  # convert to network byte order
        target_socket.send(msg)

    def send_double(self, target_socket, val):
        """8 byte, 64bit"""
        msg = struct.pack('!d', val)  # convert to network byte order
        target_socket.send(msg)

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
