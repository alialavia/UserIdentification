#!/usr/bin/python
from src.config import *


class Error:

    def __init__(self, server, conn, error_msg="An error occurred during processing of the request"):

        # send back reponse identifier
        resp_id = CONFIG['ROUTING']['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back error message
        server.send_string(conn, error_msg)
