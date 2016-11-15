#!/usr/bin/python

# import TCP server interface
from src.lib.TCPServer import TCPServerBlocking
import cv2
from time import sleep

REQUEST_LOOKUP = {
    1: 'image_handling',
    2: 'primitive_handling'
}

class TCPTestServer(TCPServerBlocking):

    def __init__(self, host, port):
        TCPServerBlocking.__init__(self, host, port)

    def handle_request(self, conn, addr):
        """general request handler"""
        request_id = self.receive_char(conn)

        if(request_id in REQUEST_LOOKUP):
            request = REQUEST_LOOKUP[request_id]
            if request_id == 1:
                print '--- '+str(request_id)+': Handling image...'
                self.handle_image(conn)
            elif request_id == 2:
                print '--- '+str(request_id)+': Handling primitive values...'
                self.handle_primitive_values(conn)
            else:
                print '--- Invalid request identifier, shutting down server...'
                self.SERVER_STATUS = -1  # shutdown server

        # communication finished - close socket
        conn.close()

    #  ----------- REQUEST HANDLERS

    def handle_primitive_values(self, conn):
        # receive char
        print "--- char: " + str(self.receive_char(conn))
        print "--- uchar: " + str(self.receive_uchar(conn))
        print "--- short: " + str(self.receive_short(conn))
        print "--- ushort: " + str(self.receive_ushort(conn))
        print "--- int: " + str(self.receive_int(conn))
        print "--- uint: " + str(self.receive_uint(conn))
        print "--- bool: " + str(self.receive_bool(conn))
        print "--- float: " + str(self.receive_float(conn))

        self.send_char(conn, 110)
        self.send_uchar(conn, 250)
        self.send_short(conn, 32766)
        self.send_ushort(conn, 65534)
        self.send_int(conn, 2147483647)
        self.send_uint(conn, 4294967000)
        self.send_bool(conn, True)
        # TODO: fix, not working atm
        self.send_float(conn, 12.0)

    def handle_image(self, conn):
        """receive image, draw and send back"""
        size = self.receive_integer(conn)
        img = self.receive_rgb_image(conn, size, size)
        height, width, channels = img.shape
        print '--- Received image'
        # draw circle in the center
        cv2.circle(img, (width/2, height/2), height/4, (0, 0, 255), -1)
        # send image back
        self.send_rgb_image(conn, img)

# ================================= #
#              Main

if __name__=='__main__':

    server = TCPTestServer('', 999)
    server.start_server()
