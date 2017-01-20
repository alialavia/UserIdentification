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

void sendTrainingBatch(io::TCPClient *c, const std::vector<cv::Mat> &image_batch)
{

	std::cout << "--- Sending " << image_batch.size() << " images to server" << std::endl;

	// --- send image dimension
	std::cout << "--- " << c->SendUInt(image_batch[0].size().width) << " bytes sent (image size)" << std::endl;

	// --- send number of images
	std::cout << "--- " << c->SendUChar(image_batch.size()) << " bytes sent (nr images)" << std::endl;

	// -- send images
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

enum Mode
{
	Mode_none = 0,
	Mode_training = 1,
	Mode_trigger_classifier_training = 2,
	Mode_embedding_calculation = 3,
	Mode_identification = 4,
};

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	io::KinectSensorMultiSource k;
	HRESULT hr;
	cv::Mat color_image;
	enum Mode MODE = Mode_none;

	// print instructions
	std::cout << "=====================================\n"
				 "          INSTRUCTIONS\n"
				 "=====================================\n"
				 "[1]: send training images - use [space] to collect face captures\n"
				 "[2]: trigger classifier training\n"
				 "[3]: direct embedding calculation - use [space] to take a picture\n"
				 "[4]: identification mode - use [space] to send face capture\n"
				 "[q]: Quit\n"
				 "\n\n";

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

	// config to server connection
	io::TCPClient c;
	c.Config("127.0.0.1", FLAGS_port);

	int nr_images = 0;
	std::vector<cv::Mat> image_batch;
	int key = (int)('-1');

	while (true)
	{
		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// mode selection
			if(MODE == Mode_none)
			{
				if ((int)('1') == key)	// space = save
				{
					MODE = Mode_training;
					std::cout << "--- Starting training mode...\n";
				}
				else if ((int)('2') == key)
				{
					MODE = Mode_trigger_classifier_training;
					std::cout << "--- Trigger identification mode...\n";
				}
				else if ((int)('3') == key)
				{
					MODE = Mode_embedding_calculation;
					std::cout << "--- Starting identification mode...\n";
				}
				else if ((int)('4') == key)
				{
					MODE = Mode_identification;
					std::cout << "--- Starting identification mode...\n";
				}
				else if ((int)('q') == key)
				{
					std::cout << "--- Terminating...\n";
					break;
				}
			}


			if(MODE == Mode_training)
			{
				
				// extract skeleton data
				IBody** bodies = k.GetBodyDataReference();
				st.ExtractJoints(bodies);

				// get face bounding boxes
				std::vector<cv::Rect2f> bounding_boxes;
				std::vector<int> user_scene_ids;
				st.GetFaceBoundingBoxesRobust(bounding_boxes, user_scene_ids, base::ImageSpace_Color);

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

							// connect to server
							if (!c.Connect())
							{
								std::cout << "Could not connect to server" << std::endl;
								return -1;
							}

							// send request ID to server
							// 2: send training images
							c.SendUChar(2);
							// send user id
							std::cout << "Please enter a user name: ";
							c.SendKeyboard();

							sendTrainingBatch(&c, image_batch);
							// reset batch
							nr_images = 0;
							image_batch.clear();
							cv::destroyWindow("Face");
							// reset control mode
							MODE = Mode_none;
							// request terminated - reconnect to server
							c.Close();
						}
					}	//	/space
				}	// /bounding boxes
			}else if(MODE == Mode_trigger_classifier_training)
			{

				// connect to server
				if (!c.Connect())
				{
					std::cout << "Could not connect to server" << std::endl;
					return -1;
				}

				// send request ID to server
				c.SendUChar(4);

				MODE = Mode_none;
				c.Close();

			}else if(MODE == Mode_embedding_calculation)
			{
				// extract skeleton data
				IBody** bodies = k.GetBodyDataReference();
				st.ExtractJoints(bodies);

				// get face bounding boxes
				std::vector<cv::Rect2f> bounding_boxes;
				std::vector<int> user_scene_ids;
				st.GetFaceBoundingBoxesRobust(bounding_boxes, user_scene_ids, base::ImageSpace_Color);

				if (bounding_boxes.size() > 0)
				{
					// take first person
					cv::Mat face = color_image(bounding_boxes[0]);

					// show image
					cv::imshow("Face", face);
					int key = cv::waitKey(5);

					if (key == 32)	// space = save
					{
						// resize
						cv::resize(face, face, cv::Size(96, 96), 0, 0);

						// connect to server
						if (!c.Connect())
						{
							std::cout << "Could not connect to server" << std::endl;
							MODE = Mode_none;
							c.Close();
							continue;
						}

						// send request ID to server
						c.SendUChar(3);
						// image size
						c.SendUInt(face.size().width);
						// send image
						std::cout << "--- sent " << c.SendRGBImage(face) << " bytes to server\n";
						cv::destroyWindow("Face");

						// ----- receive
						//int user_id = c.Receive32bit<int>();
						//float confidence = c.Receive32bit<float>();
						//std::cout << "=== DETECTED USER: " << user_id << " | confidence: " << confidence << std::endl;



						// cleanup
						cv::destroyWindow("Face");
						MODE = Mode_none;
						c.Close();
					}

					// draw bounding boxes
					st.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);
				}


			}else if(MODE == Mode_identification)
			{
				// extract skeleton data
				IBody** bodies = k.GetBodyDataReference();
				st.ExtractJoints(bodies);

				// get face bounding boxes
				std::vector<cv::Rect2f> bounding_boxes;
				std::vector<int> user_scene_ids;
				st.GetFaceBoundingBoxesRobust(bounding_boxes, user_scene_ids, base::ImageSpace_Color);

				if (bounding_boxes.size() > 0)
				{
					// take first person
					cv::Mat face = color_image(bounding_boxes[0]);
					
					// show image
					cv::imshow("Face", face);
					int key = cv::waitKey(5);

					if (key == 32)	// space = save
					{
						// resize
						cv::resize(face, face, cv::Size(96, 96), 0, 0);

						// connect to server
						if (!c.Connect())
						{
							std::cout << "Could not connect to server" << std::endl;
							return -1;
						}

						// send request ID to server
						c.SendUChar(1);
						// image size
						c.SendUInt(face.size().width);
						// send image
						std::cout << "--- sent " << c.SendRGBImage(face) << " bytes to server\n";
						cv::destroyWindow("Face");

						// ----- receive
						int user_id = c.Receive32bit<int>();
						float confidence = c.Receive32bit<float>();
						std::cout << "=== DETECTED USER: " << user_id << " | confidence: "<< confidence << std::endl;

						// cleanup
						cv::destroyWindow("Face");
						MODE = Mode_none;
						c.Close();
					}

					// draw bounding boxes
					st.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);
				}


			}

			// display image
			cv::imshow("Color Stream", color_image);
			key = cv::waitKey(5);

		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	// close camera
	k.Close();

	return 0;
}
