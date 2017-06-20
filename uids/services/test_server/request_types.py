#!/usr/bin/python
import response_types as r
from uids.utils.Logger import Logger as log
# config
from config import ROUTING
r.ROUTING = ROUTING


class Ping:
    def __init__(self, server, conn, handle):
        print "Ping from: ", conn.getpeername()
        r.Pong(server, conn)


