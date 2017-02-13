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

TCPClient::TCPClient(): mSocketID(-1), mHostName(nullptr), mHostPort(-1)
{

}

TCPClient::~TCPClient()
{
	Close();
}


void TCPClient::Config(char* host_name, int host_port)
{
	mHostName = host_name;
	mHostPort = host_port;
}


bool TCPClient::Connect()
{
	return Connect(mHostName, mHostPort);
}

bool TCPClient::Connect(char* host_name, int host_port)
{

#ifdef _DEBUG_NETWORKING
	std::cout << "=== Connecting to server: " << host_name << ":" << host_port << std::endl;
#endif

	// close open connection
	if (mSocketID != -1)
	{
		Close();
	}

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

void TCPClient::Close()
{
	closesocket(mSocketID);
	WSACleanup();
	mSocketID = -1;
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

std::string TCPClient::ReceiveStringWithVarLength()
{
	// receive char array length
	int nr_bytes = Receive32bit<int>();

	char * buffer = new char[nr_bytes];
	std::string receivedString = "";
	// receive image
	int succ = ReceiveMessage(mSocketID, buffer, &nr_bytes);

	if (succ < 0) {
		// error
	}
	else
	{
		receivedString = std::string(buffer, nr_bytes);
	}

	delete[] buffer;
	return receivedString;
}

// return -1 on failure, 0 on success
int TCPClient::ReceiveMessage(SOCKET socket, char *buf, int *len)
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

int TCPClient::ReceiveRGBImageQuadratic(cv::Mat &output)
{
	short img_dim = Receive16bit<short>();
	output = cv::Mat::zeros(img_dim, img_dim, CV_8UC3);
	int buffer_length = static_cast<int>(output.total()*output.elemSize());

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

int TCPClient::ReceiveRGBImage(cv::Mat &output)
{
	// width
	short width = Receive16bit<short>();
	// height
	short height = Receive16bit<short>();

	output = cv::Mat::zeros(height, width, CV_8UC3);
	int buffer_length = static_cast<int>(output.total()*output.elemSize());

	// allocate memory
	char * sockData = new char[buffer_length];
	// receive image
	int bytes_recv = ReceiveMessage(mSocketID, sockData, &buffer_length);
	// apply to opencv header
	cv::Mat img(cv::Size(height, width), CV_8UC3, sockData);
	// deep copy
	output = img.clone();
	// delete buffer
	delete[] sockData;
	return bytes_recv;
}

// ------ send

int TCPClient::SendChar(char id) const
{
	int bytecount;
	// send 1 byte identifier = char
	if ((bytecount = send(mSocketID, &id, sizeof(char), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendChar sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendUInt(uint32_t size) const
{
	uint32_t network_byte_order = htonl(size);
	int bytecount;
	// send 4 bytes message with datalength
	if ((bytecount = send(mSocketID, (char*)&network_byte_order, sizeof(uint32_t), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendUInt sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendInt(int size) const
{
	uint32_t network_byte_order = htonl(size);
	int bytecount;
	// send 4 bytes message with datalength
	if ((bytecount = send(mSocketID, (char*)&network_byte_order, sizeof(int), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendInt sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendShort(short val) const
{
	unsigned short network_byte_order = htons(val);
	int bytecount;
	// send 2 bytes message with datalength
	if ((bytecount = send(mSocketID, (char*)&network_byte_order, sizeof(unsigned short), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendShort sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendUShort(unsigned short ushort) const
{
	unsigned short network_byte_order = htons(ushort);
	int bytecount;
	// send 2 bytes message with datalength
	if ((bytecount = send(mSocketID, (char*)&network_byte_order, sizeof(unsigned short), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendUShort sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendBool(bool val) const
{
	int bytecount;
	// send 1 byte identifier = char
	if ((bytecount = send(mSocketID, (char*)&val, sizeof(char), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendBool sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendDouble(double val) const
{
	int bytecount;
	uint64_t network_byte_order = htond(val);

	// send 1 byte identifier = char
	if ((bytecount = send(mSocketID, (char*)&network_byte_order, sizeof(double), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendDouble sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendFloat(float val) const
{
	int bytecount;
	unsigned int network_byte_order = htonf(val);

	// send 1 byte identifier = char
	if ((bytecount = send(mSocketID, (char*)&network_byte_order, sizeof(float), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendFloat sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendUChar(unsigned char val) const
{
	int bytecount;
	// send 1 byte identifier = char
	if ((bytecount = send(mSocketID, (char*)&val, sizeof(char), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendUChar sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

int TCPClient::SendRGBImageData(const cv::Mat &img) const
{

#ifdef _DEBUG_NETWORKING
	if (img.rows == 0 || img.cols == 0) {
		std::cout << "Could not empty image.";
		return 0;
	}
#endif

	cv::Mat frame = img.clone();

	if (!frame.isContinuous())
	{
#ifdef _DEBUG_NETWORKING
		std::cout << "Image is not continuous (maybe ROI)\n";
#endif
		frame = frame.clone();
	}

	frame = (frame.reshape(0, 1)); // to make it continuous
	const int imgSize = static_cast<const int>(frame.total()*frame.elemSize());

	int bytecount;
	if ((bytecount = send(mSocketID, (const char *)frame.data, imgSize, 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}

#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendRGBImageData sent " << bytecount << "bytes\n";
#endif
	return bytecount;
}

void TCPClient::SendRGBTestImage(int size) const
{
	cv::Mat frame = cv::Mat::zeros(size, size, CV_8UC3);
	frame = (frame.reshape(0, 1)); // to make it continuous

	// send size
	const int imgSize = static_cast<const int>(frame.total()*frame.elemSize());

	int bytecount;
	if ((bytecount = send(mSocketID, (const char *)frame.data, imgSize, 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return;
	}
#ifdef _DEBUG_NETWORKING
	std::cout << "--- SendRGBTestImage sent " << bytecount << "bytes\n";
#endif
	printf("Sent bytes: %d\n", bytecount);
}

int TCPClient::SendString(std::string val) const
{

	// buffer size
	SendUInt(static_cast<int>(val.size()));

	// send cstring
	int bytecount;
	if ((bytecount = send(mSocketID, val.c_str(), static_cast<int>(val.size()), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return 0;
	}
	return bytecount;
}

bool TCPClient::SendKeyboard() const
{
	char buffer[1024];
	int buffer_len = 1024;
	int bytecount;

	int c;
	memset(buffer, '\0', buffer_len);

	// receive characters
	for (char* p = buffer; (c = _getch()) != 13; p++) {
		printf("%c", c);
		*p = c;
	}

	// buffer size
	SendUInt(static_cast<uint32_t>(strlen(buffer)));

	// send char array
	if ((bytecount = send(mSocketID, buffer, static_cast<int>(strlen(buffer)), 0)) == SOCKET_ERROR) {
		fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
		return false;
	}

	return true;
}

void TCPClient::SendImageBatchQuadraticSameSize(const std::vector<cv::Mat> &images) const
{
#ifdef _DEBUG_NETWORKING
	// check image dimensions
	int img_size = images[0].size().width;
	for (size_t i = 0; i<images.size(); i++)
	{
		if (images[i].size().width != img_size || images[i].size().height != img_size) {
			throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
		}
	}
#endif
	if (images.size() == 0)
	{
		throw std::invalid_argument("No data to send.");
	}

	// send nr images
	SendShort(static_cast<short>(images.size()));

	// send image dimension
	SendShort(images[0].size().width);

	// send images
	for (size_t i = 0; i<images.size(); i++)
	{
		SendRGBImageData(images[i]);
	}
}

void TCPClient::SendImageBatchSameSize(const std::vector<cv::Mat> &images) const
{
#ifdef _DEBUG_NETWORKING
	// check image dimensions
	int img_width = images[0].size().width;
	int img_height = images[0].size().height;
	for (size_t i = 0; i<images.size(); i++)
	{
		if (images[i].size().width != img_width || images[i].size().height != img_height) {
			throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
		}
	}
#endif
	if (images.size() == 0)
	{
		throw std::invalid_argument("No data to send.");
	}

	// send nr images
	SendShort(static_cast<short>(images.size()));

	// send image dimension
	SendShort(images[0].size().width);
	SendShort(images[0].size().height);

	// send images
	for (size_t i = 0; i<images.size(); i++)
	{
		SendRGBImageData(images[i]);
	}
}

void TCPClient::SendImageBatchQuadratic(const std::vector<cv::Mat> &images) const
{
#ifdef _DEBUG_NETWORKING
	// check image dimensions
	for (size_t i = 0; i<images.size(); i++)
	{
		if (images[i].size().width != images[i].size().height) {
			throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
		}
	}
#endif
	if (images.size() == 0)
	{
		throw std::invalid_argument("No data to send.");
	}

	// send nr images
	SendShort(static_cast<short>(images.size()));

	// send images
	for (size_t i = 0; i<images.size(); i++)
	{
		// send image dimension
		SendShort(images[i].size().width);

		// send image
		SendRGBImageData(images[i]);
	}
}

void TCPClient::SendImageBatch(const std::vector<cv::Mat> &images) const
{
	if (images.size() == 0)
	{
		throw std::invalid_argument("No data to send.");
	}

	// send nr images
	SendShort(static_cast<short>(images.size()));

	// send images
	for (size_t i = 0; i<images.size(); i++)
	{
		// send image width
		SendShort(images[i].size().width);

		// send image height
		SendShort(images[i].size().height);

		// send image
		SendRGBImageData(images[i]);
	}
}

int TCPClient::SendRGBImage(const cv::Mat &img) const {
	int data_sent = 0;
	// send image dimension
	data_sent += SendShort(img.size().width);
	data_sent += SendShort(img.size().height);
	data_sent += SendRGBImageData(img);
	return data_sent;
}

int TCPClient::SendRGBImageQuadratic(const cv::Mat &img) const {
#ifdef _DEBUG_NETWORKING
	if (img.size().width != img.size().height) {
		throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
	}
#endif
	int data_sent = 0;
	// send image dimension
	data_sent += SendShort(static_cast<short>(img.size().width));
	data_sent += SendRGBImageData(img);
	return data_sent;
}


