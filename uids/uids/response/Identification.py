#!/usr/bin/python
from uids.config import *


class Identification:

    def __init__(self, server, conn, user_id, user_name):

        # send back reponse identifier
        resp_id = CONFIG['ROUTING']['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back user id
        server.send_int(conn, int(user_id))

        # send back nice name
        server.send_string(conn, user_name)
