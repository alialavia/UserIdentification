#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>

#include <gflags/gflags.h>
#include "io/Networking.h"
#include "tracking/FaceTracker.h"

DEFINE_int32(port, 8080, "Server port");

void sendTrainingBatch(io::TCPClient *c, const std::vector<cv::Mat*> &image_batch)
{
	std::cout << "--- Sending " << image_batch.size() << " images to server" << std::endl;

	// --- send image dimension
	std::cout << "--- " << c->SendUInt((*image_batch[0]).size().width) << " bytes sent (image size)" << std::endl;

	// --- send number of images
	std::cout << "--- " << c->SendUChar(image_batch.size()) << " bytes sent (nr images)" << std::endl;

	// -- send images
	for (int i = 0; i < image_batch.size(); i++) {
		std::cout << "sent " << c->SendRGBImage(*image_batch[i]) << " bytes to server\n";
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
	Mode_identification = 3,
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
		"[1]: autocollect face snapshots - use [space] to send them to the server for training\n"
		"[2]: trigger classifier training\n"
		"[3]: identification mode - use [space] to send face capture\n"
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
	tracking::FaceTracker ft(pSensor);
	tracking::RadialFaceGrid grid;
	cv::Mat face_snap;

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
			if (MODE == Mode_none)
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
					MODE = Mode_identification;
					std::cout << "--- Starting identification mode...\n";
				}
				else if ((int)('q') == key)
				{
					std::cout << "--- Terminating...\n";
					break;
				}
			}


			if (MODE == Mode_training)
			{

				// extract raw face data
				FaceData* face_data_raw = k.GetFaceDataReference();

				// copy/convert
				ft.ExtractFacialData(face_data_raw);

				// get face bounding boxes
				std::vector<cv::Rect2f> bounding_boxes;
				ft.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);

				if (bounding_boxes.size() > 0)
				{
					face_snap = color_image(bounding_boxes[0]);
				}	// /bounding boxes

					// faces
				std::vector<tracking::Face> faces = ft.GetFaces();
				for (int i = 0; i < faces.size(); i++) {

					int roll, pitch, yaw;
					faces[i].GetEulerAngles(roll, pitch, yaw);

					try
					{
						// add face if not yet capture from this angle
						if (grid.IsFree(roll, pitch, yaw)) {
							grid.StoreSnapshot(roll, pitch, yaw, face_snap);
						}
					}
					catch (...)
					{
					}
				}

				// get face capture grid
				cv::Mat face_captures;
				grid.GetFaceGridPitchYaw(face_captures);

				// draw bounding boxes
				ft.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);
				ft.RenderFaceFeatures(color_image, base::ImageSpace_Color);

				
				// show image
				cv::imshow("Face Grid", face_captures);
				int key = cv::waitKey(3);
				if (key == 32)	// space = send training batch to server
				{

					grid.ResizeImages(100);
					std::vector<cv::Mat*> batch = grid.ExtractGrid();

					if(batch.size()>0)
					{
						// connect to server
						if (c.Connect())
						{
							// send request ID to server
							// 2: send training images
							c.SendUChar(2);
							// send user id
							std::cout << "Please enter a user name: ";
							c.SendKeyboard();

							sendTrainingBatch(&c, batch);
							grid.Clear();

							MODE = Mode_none;
							c.Close();
						}else
						{
							std::cout << "Could not connect to server. Please try again" << std::endl;
						}

					}else
					{
						std::cout << "Please wait till more snapshots are collected.";
					}
				}
		

			}
			else if (MODE == Mode_trigger_classifier_training)
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

			}
			else if (MODE == Mode_identification)
			{
			
	

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
