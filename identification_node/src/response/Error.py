#!/usr/bin/python
from src.config import *


class Error:

    def __init__(self, server, conn, error_mgs):

        resp_id = CONFIG['ROUTING']['RESPONSE']['ID']['Error']

        # send back user id
        server.send_int(conn, int(resp_id))

        # send back error message
        server.send_string(conn, error_mgs)
