import socket  # for sockets
import sys  # for exit

try:
    # create an AF_INET, STREAM socket (TCP)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error, msg:
    print 'Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + msg[1]
    sys.exit();

print 'Socket Created'

# docker container is listening to port 80
clientsocket.connect(('localhost', 80))

try:
    # Set the whole string
    clientsocket.send('Hi there! I\'m your Client :)')
except socket.error:
    # Send failed
    print 'Send failed'
    sys.exit()

print 'Message send successfully'

# Now receive data
reply = clientsocket.recv(4096)

print 'Received from server: ' + reply