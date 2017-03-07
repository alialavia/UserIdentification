#!/usr/bin/env python2
import argparse
import importlib            # msg routing imports
from config import *    # server configuration
from uids.UserDB import UserDB   # user database
from uids.features.EmbeddingGen import EmbeddingGen   # CNN embedding generator
from uids.networking.TCPServer import TCPServerBlocking # tcp networking
from uids.utils.Logger import Logger as log
from uids.online_learning.BatchProcessing import BatchProcessingMultiClassTree

# import request types
M_REQUESTS = importlib.import_module("request_types")


class IdentificationServer(TCPServerBlocking):

    classifier = None
    user_db = None
    embedding_gen = None

    def __init__(self, host, port):
        TCPServerBlocking.__init__(self, host, port)

        # CNN generator
        self.embedding_gen = EmbeddingGen()

        # User DB
        self.user_db = UserDB()

        # Classifier - linked to database
        # self.classifier = OfflineMultiClassTree(self.user_db, 'OCSVM')
        self.classifier = BatchProcessingMultiClassTree(self.user_db)

    def handle_request(self, conn, addr):
        """general request handler"""
        request_id = self.receive_uchar(conn)

        # message routing
        req_lookup = ROUTING['REQUEST']['NAME']

        if request_id in req_lookup:
            req_type = req_lookup[request_id]

            try:
                req = getattr(M_REQUESTS, req_type)
                # handle request
                req(self, conn)
            except AttributeError:
                log.error("Request model '"+req_type+"' is not yet implemented or an Exception occurred.")
        else:
            log.error("Unsupported request type: " + str(request_id))

        # communication finished - close connection

# ================================= #
#              Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help="Server port.", default=8080)

    args = parser.parse_args()

    server = IdentificationServer('', args.port)
    server.start_server(verbose=False)
