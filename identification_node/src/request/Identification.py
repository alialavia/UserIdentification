#!/usr/bin/python
from src.config import *


class Identification:

    def __init__(self, server, conn):
        print "Identification request"

        print "--- Identification"

        # receive image size
        img_size = server.receive_uint(conn)
        # receive image
        user_face = server.receive_rgb_image(conn, img_size, img_size)
        # identify
        user_id, nice_name, confidence = server.classifier.identify_user(user_face)

        if (user_id is None):
            print "--- Identification Error"
            # send back error response
            server.send_int(conn, 99)
            return

        print "--- User ID: " + str(user_id) + " | confidence: " + str(confidence)

        # send back response type
        server.send_int(conn, 1)

        # send back user id
        server.send_int(conn, int(user_id))

        # send back nice name
        server.send_string(conn, nice_name)

        # send back confidence
        server.send_float(conn, float(confidence))
