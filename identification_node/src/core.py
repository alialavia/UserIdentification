'''A sample of Python TCP server'''

import socket

# HOST = '127.0.0.1'     # Local host
HOST = 'localhost'
PORT = 80              # expose when starting the container!

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))

# maximum number of connectsion
s.listen(1)

print 'Waiting for connection...'
conn, addr = s.accept()

print 'Connected by client', addr
while True:

    # Wait for a connection
    connect, address = s.accept()

    # Typically fork at this point

    # Receive up to 1024 bytes
    resp = (connect.recv(1024)).strip()
    # And if the user has sent a "SHUTDOWN"
    # instruction, do so (ouch! just a demo)
    if resp == "SHUTDOWN": break

    # Send an answer
    connect.send("You said '" + resp + "' to me\n")

conn.close()
print 'Server closed.'