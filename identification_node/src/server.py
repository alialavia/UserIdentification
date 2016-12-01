#!/usr/bin/env python2
import argparse
from src.config import *    # server configuration
import importlib            # msg routing imports
from src.lib.UserDB import UserDB   # user database
from src.lib.EmbeddingGen import EmbeddingGen   # CNN embedding generator
from src.lib.SVM import SVM     # SVM classifier
from src.lib.TCPServer import TCPServerBlocking # tcp networking
# import request types
M_REQUESTS = importlib.import_module("src.request")


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
        self.classifier = SVM(self.user_db)

    def handle_request(self, conn, addr):
        """general request handler"""
        request_id = self.receive_uchar(conn)

        # message routing
        req_lookup = CONFIG['ROUTING']['REQUEST']['NAME']

        if request_id in req_lookup:
            req_type = req_lookup[request_id]
            try:
                req_module = getattr(M_REQUESTS, req_type)
                req = getattr(req_module, req_type)
                # handle request
                req(self, conn)
            except AttributeError:
                print "Request model is not yet implemented."
        else:
            print "Unsupported request type: " + str(request_id)

        # communication finished - close connection
        conn.close()

# ================================= #
#              Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help="Server port.", default=8080)

    args = parser.parse_args()

    server = IdentificationServer('', args.port)
    server.start_server()
