#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/cvdef.h>
#include "io/KinectInterface.h"

DEFINE_int32(port, 80, "Server port");
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

void receive_primitive(io::TCPClient *c)
{
	// send request ID
	c->SendChar(2);
	std::cout << "--- Server responded: " << c->ReceiveUnsignedShortInt() << std::endl;
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
			receive_primitive(&c);
		}else if(FLAGS_message_type == "image")
		{
			send_and_receive_image(&c);
		} else
		{
			std::cout << "--- invalid message type\n";
			break;
		}

		Sleep(1000);

		c.Reconnect();
		// connection is terminated by server
	}

	return 0;
} 