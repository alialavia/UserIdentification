#include <winsock2.h>

#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <conio.h>
#include "io/Networking.h"

#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>


using namespace io;

// ----------------------------- SOCKETS

TCPClient::TCPClient()
{
	
}

TCPClient::~TCPClient()
{
	Close();
}

bool TCPClient::Connect(char* host_name, int host_port)
{
	// open new socket
	OpenSocket();

	struct sockaddr_in my_addr;
	mHostName = host_name;
	mHostPort = host_port;

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

void TCPClient::Reconnect()
{

	std::cout << "=== Reconnecting to server" << std::endl;
	// existing socket if not yet done
	if(mSocketID != -1)
	{
		Close();
	}

	// reconnect
	Connect(mHostName, mHostPort);
}

void TCPClient::Close()
{
	closesocket(mSocketID);
	WSACleanup();
	mSocketID = -1;
}

unsigned int TCPClient::ReceiveUnsignedInt()
{
	// id is a 4 byte/32 bit integer
	long rc;
	unsigned int nr;
	rc = recv(mSocketID, (char*)&nr, sizeof(unsigned int), 0);
	return ntohl(nr);
}
unsigned short int TCPClient::ReceiveUnsignedShortInt()
{
	// id is a 2 byte/16 bit integer
	long rc;
	unsigned short int nr;
	rc = recv(mSocketID, (char*)&nr, sizeof(unsigned short int), 0);
	return ntohs(nr);
}

// return -1 on failure, 0 on success
int TCPClient::ReceiveMessage(int socket, char *buf, int *len)
{
	int total = 0;        // how many bytes we've received
	int bytesleft = *len; // how many we have left to receive
	int n = -1;

	while (total < *len) {
		n = recv(socket, buf + total, bytesleft, 0);
		if (n <= 0) { break; }
		total += n;
		bytesleft -= n;
	}

	*len = total; // return number actually received here

	return (n <= 0) ? -1 : 0;
}

int TCPClient::ReceiveRGBImage(cv::Mat &output, int img_dim)
{
	output = cv::Mat::zeros(img_dim, img_dim, CV_8UC3);
	int  buffer_length = output.total()*output.elemSize();

	// allocate memory
	char * sockData = new char[buffer_length];
	// receive image
	int bytes_recv = ReceiveMessage(mSocketID, sockData, &buffer_length);
	// apply to opencv header
	cv::Mat img(cv::Size(img_dim, img_dim), CV_8UC3, sockData);
	// deep copy
	output = img.clone();
	// delete buffer
	delete[] sockData;
	return bytes_recv;
}

int TCPClient::SendChar(char id)
{
	int bytecount;
	// send 1 byte identifier = char
	if ((bytecount = send(mSocketID, &id, sizeof(char), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
	return bytecount;
}

int TCPClient::SendUInt(uint32_t size)
{
	uint32_t network_byte_order;
	network_byte_order = htonl(size);
	int bytecount;
	// send 4 bytes message with datalength
	if ((bytecount = send(mSocketID, (char*)&network_byte_order, sizeof(unsigned int), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
	return bytecount;
}

int TCPClient::SendImageWithLength(const cv::Mat &img)
{
	//cv::Mat frame = cv::Mat::zeros(size, size, CV_8UC3);
	cv::Mat flattend= (img.reshape(0, 1)); // to make it continuous
	const int imgSize = flattend.total()*flattend.elemSize();

	// first send message size
	SendUInt(imgSize);

	// then send flattened image
	int bytecount;
	if ((bytecount = send(mSocketID, (const char *)flattend.data, imgSize, 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
	return bytecount;
}


int TCPClient::SendImageBatchWithLength(const std::vector<cv::Mat> &images)
{
	// send number of images
	SendChar(images.size());

	// send image size
	const int imgSize = images[0].cols * images[0].rows * images[0].elemSize();
	SendUInt(imgSize);

	// send images
	int bytecount = 0, totalbytecount = 0;
	for(int i = 0; i<images.size();i++)
	{
		cv::Mat flattend = (images[i].reshape(0, 1));
		// send data
		if ((bytecount = send(mSocketID, (const char *)flattend.data, imgSize, 0)) == SOCKET_ERROR) {
			fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
			return totalbytecount;
		}
		totalbytecount += bytecount;
	}

	return totalbytecount;
}

int TCPClient::SendRGBImage(const cv::Mat &img)
{
	cv::Mat frame = (img.reshape(0, 1)); // to make it continuous
	const int imgSize = frame.total()*frame.elemSize();

	int bytecount;
	if ((bytecount = send(mSocketID, (const char *)frame.data, imgSize, 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
	return bytecount;
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
	printf("Sent bytes: %d\n", bytecount);
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
