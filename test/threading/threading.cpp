#include <iostream>
#include <Windows.h>
#include <opencv2/core.hpp>
#include "gflags/gflags.h"

#include "io/KinectInterface.h"
#include "io/Networking.h"
#include <io/RequestHandler.h>
#include <io/RequestTypes.h>
#include <io/ResponseTypes.h>

typedef io::IdentificationRequestSingleImage req;


DEFINE_int32(port, 9999, "Server port");
int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	
	// initialize sensor
	//io::KinectSensorMultiSource k;
	//if (FAILED(k.Open())) {
	//	std::cout << "--- Initialization failed" << std::endl;
	//	return -1;
	//}

	// connect to server
	io::TCPClient c;
	if (!c.Connect("127.0.0.1", FLAGS_port))
	{
		std::cout << "--- Could not connect to server" << std::endl;
		return -1;
	}

	// start request handler
	io::NetworkRequestHandler req_handler;
	req_handler.start(); // parallel processing

	// allocate state data
	cv::Mat color_image;
	HRESULT hr;

	while (true)
	{
		// polling
		//hr = k.AcquireFrame();
		hr = 1;

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			///k.GetImageCopyRGB(color_image);

			color_image = cv::Mat::zeros(96, 96, CV_8UC3);

			cv::resize(color_image, color_image, cv::Size(96, 96), 0, 0);

			// handle processed requests
			io::IdentificationResponse response;
			while(req_handler.PopResponse(&response))
			{
				// display response
				std::cout << response.mUserID << std::endl;
			}

			// make new request
			req* new_request =  new req(&c, color_image);
			req_handler.addRequest(new_request);

			// simulate delay

			Sleep(1000);
		}

	}


	// blocking call - wait till request handler is finished (processRequests terminates)
	req_handler.stop(); 
}
