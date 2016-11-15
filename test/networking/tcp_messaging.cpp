#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/cvdef.h>
#include "io/KinectInterface.h"

DEFINE_int32(port, 8080, "Server port");
DEFINE_string(message_type, "primitive", "message types: image, primitive");

void send_and_receive_image(io::TCPClient *c)
{
	// send request ID
	c->SendChar(1);
	// send image
	c->SendRGBTestImage(100);
	std::cout << "--- image sent, now waiting to receive\n";
	// receive image
	cv::Mat server_img = cv::Mat::zeros(100, 100, CV_8UC3);
	c->ReceiveRGBImage(server_img, 100);
	// display image
	cv::imshow("Received from server", server_img);
	cv::waitKey(0);
}


int main(int argc, char** argv)
{

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// connect to server
	io::TCPClient c;

	if (!c.Connect("127.0.0.1", FLAGS_port))
	{
		std::cout << "=== Could not connect to server" << std::endl;
		return -1;
	}

	while(1)
	{
		if(FLAGS_message_type == "primitive")
		{

			// send request id
			c.SendChar(2);

			// ----- send
			// char
			c.SendChar(100);
			// unsigned char
			c.SendUChar(255);
			// short
			c.SendShort(32767);
			// unsigned short
			c.SendUShort(65535);
			// integer
			c.SendInt(234234);
			// unsigned integer
			c.SendUInt(4294967001);
			// bool
			c.SendBool(true);
			// float
			c.SendFloat(10.1f);

			// ----- receive
			std::cout << "int8: " << (int)c.Receive8bit<int8_t>() << std::endl;
			std::cout << "uint8: " << (int)c.Receive8bit<uint8_t>() << std::endl;
			std::cout << "short: " << c.Receive16bit<short>() << std::endl;
			std::cout << "ushort: " << c.Receive16bit<unsigned short>() << std::endl;
			std::cout << "int: " << c.Receive32bit<int>() << std::endl;
			std::cout << "uint: " << c.Receive32bit<unsigned int>() << std::endl;
			std::cout << "bool: " << c.Receive8bit<bool>() << std::endl;
			// TODO: fix
			std::cout << "float: " << c.Receive32bit<float>() << std::endl;

			// close connection
			c.Close();
			break;
		}else if(FLAGS_message_type == "image")
		{
			send_and_receive_image(&c);
			// reconnect to server
			c.Close();
			break;
		}
		// connection is terminated by server
	}

	

	return 0;
} 