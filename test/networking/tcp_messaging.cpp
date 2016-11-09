#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/cvdef.h>
#include "io/KinectInterface.h"

DEFINE_int32(port, 80, "Server port");

int main(int argc, char** argv)
{

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	io::TCPClient c;
	c.Connect("127.0.0.1", FLAGS_port);

	// send request ID
	c.SendChar(1);

	// send image
	c.SendRGBTestImage(100);
	std::cout << "--- image sent, now waiting to receive\n";
	// receive image
	cv::Mat server_img = cv::Mat::zeros(100, 100, CV_8UC3);
	c.ReceiveRGBImage(server_img, 100);
	// display image
	cv::imshow("Received from server", server_img);
	cv::waitKey(0);

} 