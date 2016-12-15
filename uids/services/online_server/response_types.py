#!/usr/bin/python

"""
Note:
    The constant "ROUTING" must be defined for this module after the import.
"""

class OK:

    def __init__(self, server, conn, msg="Request successfully processed"):

        # send back reponse identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back message
        server.send_string(conn, msg)

class Identification:

    def __init__(self, server, conn, user_id, user_name):

        # send back reponse identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back user id
        server.send_int(conn, int(user_id))

        # send back nice name
        server.send_string(conn, user_name)

class Error:

    def __init__(self, server, conn, error_msg="An error occurred during processing of the request"):

        # send back reponse identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back error message
        server.send_string(conn, error_msg)

class Embedding:

    def __init__(self, server, conn, embedding):

        # send back reponse identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back embedding array
