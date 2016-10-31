#include <winsock2.h>
#include <windows.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <conio.h>
#include "io/Networking.h"

#include <opencv2/core.hpp>

using namespace io;

TCPClient::TCPClient()
{
	OpenSocket();
}

TCPClient::~TCPClient()
{
	Close();
}

bool TCPClient::Connect(char* host_name, int host_port) const
{
	struct sockaddr_in my_addr;

	my_addr.sin_family = AF_INET;
	my_addr.sin_port = htons(host_port);

	memset(&(my_addr.sin_zero), 0, 8);
	my_addr.sin_addr.s_addr = inet_addr(host_name);

	if (connect(mSocketID, (struct sockaddr*)&my_addr, sizeof(my_addr)) == SOCKET_ERROR) {
		fprintf(stderr, "Error connecting socket %d\n", WSAGetLastError());
		return false;
	}

	return true;
}

void TCPClient::Close()
{
	closesocket(mSocketID);
	WSACleanup();
	mSocketID = -1;
}

void TCPClient::SendRGBTestImage(int size)
{
	cv::Mat frame = cv::Mat::zeros(size, size, CV_8UC3);
	frame = (frame.reshape(0, 1)); // to make it continuous

	const int imgSize = frame.total()*frame.elemSize();

	int bytecount;
	if ((bytecount = send(mSocketID, (const char *)frame.data, imgSize, 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return;
	}
	printf("Sent bytes %d\n", bytecount);
}

bool TCPClient::SendKeyboard()
{

	char buffer[1024];
	int buffer_len = 1024;
	int bytecount;

	int c;
	memset(buffer, '\0', buffer_len);

	for (char* p = buffer; (c = getch()) != 13; p++) {
		printf("%c", c);
		*p = c;
	}

	if ((bytecount = send(mSocketID, buffer, strlen(buffer), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return false;
	}
	printf("Sent bytes %d\n", bytecount);

	if ((bytecount = recv(mSocketID, buffer, buffer_len, 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error receiving data %d\n", WSAGetLastError());
		return false;
	}
	printf("Recieved bytes %d\nReceived string \"%s\"\n", bytecount, buffer);

	return true;
}

bool TCPClient::OpenSocket()
{
	// initialize socket
	unsigned short wVersionRequested;
	WSADATA wsaData;
	int err;
	wVersionRequested = MAKEWORD(2, 2);
	err = WSAStartup(wVersionRequested, &wsaData);

	if (err != 0 || (LOBYTE(wsaData.wVersion) != 2 ||
		HIBYTE(wsaData.wVersion) != 2)) {
		fprintf(stderr, "Could not find useable socket dll %d\n", WSAGetLastError());
		return false;
	}

	// initialize sockets and set any options
	int * p_int;
	mSocketID = socket(AF_INET, SOCK_STREAM, 0);
	if (mSocketID == -1) {
		printf("Error initializing socket %d\n", WSAGetLastError());
		return false;
	}

	p_int = (int*)malloc(sizeof(int));
	*p_int = 1;
	if ((setsockopt(mSocketID, SOL_SOCKET, SO_REUSEADDR, (char*)p_int, sizeof(int)) == -1) ||
		(setsockopt(mSocketID, SOL_SOCKET, SO_KEEPALIVE, (char*)p_int, sizeof(int)) == -1)) {
		printf("Error setting options %d\n", WSAGetLastError());
		free(p_int);
		return false;
	}
	free(p_int);

	return true;
}

