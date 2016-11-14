#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include "io/Networking.h"

DEFINE_int32(port, 8080, "Server port");
DEFINE_int32(batch_size, 1, "Number of images in a batch");

void sendTrainingBatch(io::TCPClient *c, int16_t user_id, const std::vector<cv::Mat> &image_batch)
{

	std::cout << "--- Sending " << image_batch.size() << " images to server" << std::endl;

	std::cout << "--- " << c->SendChar(user_id) << " bytes sent (user id)";

	// send image size
	std::cout << "--- " << c->SendUInt(image_batch[0].size().width) << " bytes sent (image size)";

	// send number of images
	std::cout << "--- " << c->SendChar(image_batch.size()) << " bytes sent (nr images)";

	for (int i = 0; i < image_batch.size(); i++) {
		std::cout << "sent " << c->SendRGBImage(image_batch[i]) << " bytes to server\n";
	}

	std::cout << "--- Image batch has been sent" << std::endl;
};

int inputUserID() {
	// How to get a number.
	int myNumber = 0;
	std::string input = "";

	while (true) {
		std::cout << "--- Please enter a valid user id >= 0: ";
		std::getline(std::cin, input);

		// This code converts from string to number safely.
		std::stringstream myStream(input);
		if (myStream >> myNumber)
			break;
		std::cout << "--- Invalid number, please try again" << std::endl;
	}
	std::cout << "--- You entered: " << myNumber << std::endl << std::endl;

	return myNumber;
};


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
	if (!c.Connect("127.0.0.1", FLAGS_port))
	{
		std::cout << "Could not connect to server" << std::endl;
		return -1;
	}

	int nr_images = 0;
	std::vector<cv::Mat> image_batch;

	while (true)
	{
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
			std::vector<int> user_scene_ids;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);
			st.GetUserSceneIDs(user_scene_ids);

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
					cv::resize(face, face, cv::Size(96, 96), 0, 0);
					image_batch.push_back(face);
					nr_images++;

					// send to server
					if (nr_images == FLAGS_batch_size) {
						// stop recording
						std::cout << "--- Captured " << nr_images << " images" << std::endl;
						int user_id = inputUserID();
						// send request ID to server
						// 2: send training images
						c.SendChar(2);
						sendTrainingBatch(&c, user_id, image_batch);
						// reset batch
						nr_images = 0;
						image_batch.clear();
						// request terminated - reconnect to server
						if (!c.Reconnect())
						{
							std::cout << "Could not reconnect to server - terminating..." << std::endl;
							return -1;
						}
					}
				}	//	/space
			}	// /bounding boxes
		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	// close camera
	k.Close();

	return 0;
}
