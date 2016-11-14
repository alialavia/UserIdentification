#!/usr/bin/python

# import TCP server interface
from src.lib.TCPServer import TCPServerBlocking
import cv2
from time import sleep

REQUEST_LOOKUP = {
    1: 'image_handling',
    2: 'binary_handling'
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
                sleep(3)  # sleep 1 second
                self.handle_primitive_values(conn)
                # simulate processing

            else:
                print '--- Invalid request identifier, shutting down server...'
                self.SERVER_STATUS = -1  # shutdown server

        # communication finished - close socket
        conn.close()

    #  ----------- REQUEST HANDLERS

    def handle_primitive_values(self, conn):
        self.send_unsigned_short(conn, 1337)

    def handle_image(self, conn):
        """receive image, draw and send back"""
        img = self.receive_rgb_image(conn, 96, 96)
        height, width, channels = img.shape
        print '--- Received image'
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
