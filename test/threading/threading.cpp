#include <iostream>
#include <Windows.h>
#include <opencv2/core.hpp>
#include "gflags/gflags.h"

#include "io/KinectInterface.h"
#include "io/Networking.h"
#include <io/RequestHandler.h>

typedef io::IdentificationRequestSingleImage req;


DEFINE_int32(port, 8080, "Server port");
int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	
	// initialize sensor
	io::KinectSensorMultiSource k;
	if (FAILED(k.Open())) {
		std::cout << "--- Initialization failed" << std::endl;
		return -1;
	}

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
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// handle processed requests

			// make new request
			req* new_request =  new req(&c, color_image);

			// submit request
			req_handler.addRequest(new_request);

		}

	}


	// blocking call - wait till request handler is finished (processRequests terminates)
	req_handler.stop(); 
}
