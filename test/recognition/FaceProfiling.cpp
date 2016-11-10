#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>

#include <io/Networking.h>

DEFINE_string(output, "output", "Output path");
DEFINE_int32(port, 555, "Server port");

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	io::KinectSensorMultiSource k;
	HRESULT hr;
	cvNamedWindow("Face", CV_WINDOW_AUTOSIZE);

	cv::Mat color_image;

	// initialize sensor
	if (FAILED(k.Open())) {
		std::cout << "Initialization failed" << std::endl;
		return -1;
	}

	// skeleton tracker
	IKinectSensor* pSensor = nullptr;

	if (FAILED(k.GetSensorReference(pSensor)))
	{
		std::cout << "Sensor is not initialized" << std::endl;
		return -1;
	}

	// init tracker
	tracking::SkeletonTracker st(pSensor);
	st.Init();

	// connect to server
	io::TCPClient c;
	if(!c.Connect("127.0.0.1", FLAGS_port))
	{
		std::cout << "Could not connect to server" << std::endl;
		return -1;
	}

	// send request ID
	c.SendChar(1);

	while (true) {

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// extract skeleton data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			// get face bounding boxes
			std::vector<cv::Rect2f> bounding_boxes;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);

			if (bounding_boxes.size() > 0)
			{
				// take first person
				cv::Mat face = color_image(bounding_boxes[0]);

				// show image
				cv::imshow("Face", face);
				int key = cv::waitKey(3);

				if (key == 32)	// space = save
				{

					// resize
					cv::resize(face, face, cv::Size(96,96), 0, 0);

					std::cout << "--- Sending image to server" << std::endl;
					// send to server
					std::cout << "sent " << c.SendRGBImage(face) << " bytes to server";
					// send only one image
					break;
				}
			}
		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	std::cout << "--- Waiting for response from server" << std::endl;

	// receive image
	cv::Mat server_img = cv::Mat::zeros(96, 96, CV_8UC3);
	c.ReceiveRGBImage(server_img, 96);
	// display image
	cv::imshow("Received from server", server_img);
	cv::waitKey(0);

	return 0;
}
