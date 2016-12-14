#!/usr/bin/env python2
import argparse
import importlib            # msg routing imports
from uids.config import *    # server configuration

from uids.lib.UserDB import UserDB   # user database
from uids.lib.EmbeddingGen import EmbeddingGen   # CNN embedding generator
from uids.lib.OfflineLearning.SVM import SVM     # SVM classifier
from uids.lib.TCPServer import TCPServerBlocking # tcp networking
# import request types
M_REQUESTS = importlib.import_module("uids.request")


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
            print("=== Incomming request: "+req_type)

            try:
                req_module = getattr(M_REQUESTS, req_type)
                req = getattr(req_module, req_type)
                # handle request
                req(self, conn)
            except AttributeError:
                print ("--- Request model '"+req_type+"' is not yet implemented.")

        else:
            print "--- Unsupported request type: " + str(request_id)

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
