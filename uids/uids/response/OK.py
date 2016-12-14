#!/usr/bin/python
from uids.config import *


class OK:

    def __init__(self, server, conn, msg="Request successfully processed"):

        # send back reponse identifier
        resp_id = CONFIG['ROUTING']['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back message
        server.send_string(conn, msg)
