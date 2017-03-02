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


class Pong:

    def __init__(self, server, conn):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))
