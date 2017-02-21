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


class UpdateFeedback:

    def __init__(self, server, conn, confidence=100):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send update confidence (uint8)
        server.send_uchar(conn, confidence)

class Reidentification:

    def __init__(self, server, conn):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))


class Error:

    def __init__(self, server, conn, error_msg="An error occurred during processing of the request"):

        # send back reponse identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back error message
        server.send_string(conn, error_msg)


class Embedding:

    def __init__(self, server, conn, embedding):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back embedding array
        raise Exception("Embedding response is not fully implemented yet.")


class QuadraticImage:

    def __init__(self, server, conn, img):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send image
        server.send_rgb_image_squared(conn, img)


class ProfilePictures:

    def __init__(self, server, conn, user_ids, pictures):

        # send back response identifier
        resp_id = ROUTING['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send number of users
        server.send_int(conn, len(user_ids))

        if len(user_ids) > 0:
            # send user ids
            for id in user_ids:
                server.send_int(conn, int(id))
            # send images
            server.send_image_batch_squared_same_size(conn, pictures)
