#!/usr/bin/python

# import TCP server interface
from src.lib.TCPServer import TCPServer

REQUEST_LOOKUP = {
    1: 'identification',
    2: 'offline_training',
    3: 'online_training'
}

class TCPTestServer(TCPServer):

    def __init__(self, host, port):
        TCPServer.__init__(self, host, port)

    def handle_request(self, conn, addr):

        # wait to receive request id
        request_id = self.receive_char(conn)

        print '--- Request ID: ' + str(request_id)

        message_length = self.receive_integer(conn)

        print '--- Message of length ' + str(message_length) + " received."

        print '--- Sending ID to client'
        self.send_unsigned_short(conn, 4)

        if(request_id in REQUEST_LOOKUP):
            request = REQUEST_LOOKUP[request_id]
            if request == 'offline_training':
                print '--- Do offline training'
            elif request == 'online_training':
                print '--- Do online training'
            elif request == 'identification':
                print '--- Do identification'
            else:
                print '--- Request Handling not yet implemented for: '.request
        else:
            print '--- Invalid request identifier, shutting down server...'
            self.SERVER_STATUS = -1     # shutdown server

        # communication finished - close connection
        conn.close()

# ================================= #
#              Main

if __name__=='__main__':

    server = TCPTestServer('', 555)
    server.start_server()
