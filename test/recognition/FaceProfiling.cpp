#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>

#include <io/Networking.h>

DEFINE_string(output, "output", "Output path");
DEFINE_int32(port, 8080, "Server port");
DEFINE_int32(batch_size, 1, "Number of images in a batch");

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

	int nr_images = 0;
	std::vector<cv::Mat> image_batch;

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
					image_batch.push_back(face);

					nr_images++;

					if (nr_images == FLAGS_batch_size) {
						// stop recording
						std::cout << "--- Captured "<< nr_images << " images" << std::endl;
						break;
					}
					
				}
			}
		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	// close camera
	c.Close();

	if (image_batch.size() == 0) {
		std::cout << "--- No images captured. Shutdown Client." << std::endl;
		return -1;
	}

	std::cout << "--- Sending image batch to server" << std::endl;

	// send image size
	c.SendUInt(image_batch[0].cols);

	// send number of images
	c.SendChar(1);

	for (int i = 0; i < image_batch.size();i++) {
		std::cout << "sent " << c.SendRGBImage(image_batch[i]) << " bytes to server\n";
	}

	std::cout << "--- Image batch has been sent" << std::endl;

	//// receive image
	//cv::Mat server_img = cv::Mat::zeros(96, 96, CV_8UC3);
	//c.ReceiveRGBImage(server_img, 96);
	//// display image
	//cv::imshow("Received from server", server_img);
	//cv::waitKey(0);

	return 0;
}
