#!/usr/bin/python
from src.config import *


class Embedding:

    def __init__(self, server, conn, embedding):

        # send back reponse identifier
        resp_id = CONFIG['ROUTING']['RESPONSE']['ID'][self.__class__.__name__]
        server.send_int(conn, int(resp_id))

        # send back embedding array
