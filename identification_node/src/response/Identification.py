#!/usr/bin/python
from src.config import *


class Identification:

    def __init__(self, server, conn, user_id, user_name):

        resp_id = CONFIG['ROUTING']['RESPONSE']['ID']['Identification']

        # send back response type
        server.send_int(conn, 1)

        # send back user id
        server.send_int(conn, int(user_id))

        # send back nice name
        server.send_string(conn, user_name)