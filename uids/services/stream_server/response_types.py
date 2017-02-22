#!/usr/bin/python

"""
Note:
    The constant "ROUTING" must be defined for this module after the import.
"""


class OK:

    def __init__(self, server, conn, msg="Request successfully processed"):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back message
        server.send_string(conn, msg)


class Identification:

    def __init__(self, server, conn, user_id, user_name, confidence=100, profile_picture=None):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back user id
        server.send_int(conn, int(user_id))

        # send back nice name
        server.send_string(conn, user_name)

        # confidence value
        server.send_uchar(conn, confidence)

        # profile picture
        if profile_picture is None:
            server.send_bool(conn, False)
        else:
            server.send_bool(conn, True)
            # send image
            server.send_rgb_image_squared(conn, profile_picture)


class UpdateFeedbackDetailed:

    def __init__(self, server, conn, user_ids, classifier_hits, max_val):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # maximum value
        server.send_uchar(conn, max_val)
        # number of classes
        server.send_int(conn, len(user_ids))
        # send classifier hits
        for i, id in enumerate(user_ids):
            server.server.send_int(conn, id)
            server.send_uchar(conn, classifier_hits[i])

class Error:

    def __init__(self, server, conn, error_msg="An error occurred during processing of the request"):

        # send back reponse identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back error message
        server.send_string(conn, error_msg)

