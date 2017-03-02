#!/usr/bin/env python2
import argparse
import importlib            # msg routing imports
from config import *    # server configuration
from uids.features.EmbeddingGen import EmbeddingGen   # CNN embedding generator
from uids.networking.TCPServer import TCPServerBlocking # tcp networking
from uids.utils.Logger import Logger as log
import time

# import request types
M_REQUESTS = importlib.import_module("request_types")


class TestServer(TCPServerBlocking):

    classifier = None
    user_db = None
    embedding_gen = None
    # message routing
    req_lookup = ROUTING['REQUEST']['NAME']

    def __init__(self, host, port):
        TCPServerBlocking.__init__(self, host, port)

        # CNN generator
        self.embedding_gen = EmbeddingGen()

    def handle_request(self, conn, addr):
        """general request handler"""

        request_id = self.receive_uchar(conn)

        # message routing
        if request_id in self.req_lookup:
            req_type = self.req_lookup[request_id]
            # log.info('server', "Incomming request: " + req_type)
            try:
                req = getattr(M_REQUESTS, req_type)
                # handle request
                req(self, conn)
            except AttributeError:
                log.error("Request model '"+req_type+"' is not yet implemented or an Exception occurred.")
        else:
            log.error("Unsupported request type: " + str(request_id))

        # request processed
        return True

# ================================= #
#              Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help="Server port.", default=8080)
    parser.add_argument('--single_request', dest='single_req_per_conn', help="Reconnect after each request.", action='store_true')
    parser.add_argument('--multi_request', dest='single_req_per_conn', action='store_false')
    parser.set_defaults(single_req_per_conn=True)

    args = parser.parse_args()

    args.single_req_per_conn = False

    server = TestServer('', args.port)
    server.start_server(one_req_per_conn=args.single_req_per_conn, verbose=args.single_req_per_conn is True)
