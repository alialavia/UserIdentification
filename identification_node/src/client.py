import socket

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# docker container is listening to port 80
clientsocket.connect(('127.0.0.1', 80))
clientsocket.send('Hi there! I\'m your Client :)')