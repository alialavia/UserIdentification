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
DEFINE_bool(send_batch, false, "send image batch to server");
DEFINE_bool(identification, true, "user identification");


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


enum IdentificationStatus
{
	IDStatus_Unknown = 0,
	IDStatus_Identified = 1,
	IDStatus_Pending = 2
};

class User
{

public:
	User() : mUserID(-1), mIDStatus(IDStatus_Unknown)
	{

	}

	~User()
	{

	}
	void SetUserID(int id)
	{
		mUserID = id;
		mIDStatus = IDStatus_Identified;
	}
	void SetIDStatus(enum IdentificationStatus status)
	{
		mIDStatus = status;
	}
	void SetFaceBoundingBox(cv::Rect2f bb)
	{
		mFaceBoundingBox = bb;
	}


	enum IdentificationStatus GetIDStatus()
	{
		return mIDStatus;
	}
	int GetUserID()
	{
		return mUserID;
	}
	cv::Rect2f GetFaceBoundingBox()
	{
		return mFaceBoundingBox;
	}


private:
	int mUserID;
	enum IdentificationStatus mIDStatus;
	cv::Rect2f mFaceBoundingBox;

};

/*
In each loop:
- refresh tracked users
- update user ids (from processed requests)
- request identification for unknown users


 */

class UserManager
{
public: 
	UserManager(): pServerConn(nullptr)
	{
	}
	bool Init(io::TCPClient* server)
	{
		if(server == nullptr)
		{
			return false;
		}
		pServerConn = server;
		return true;
	}

	// refresh tracked users: scene_id, bounding boxes
	void RefreshTrackedUsers(std::vector<int> user_scene_ids)
	{
		// update existing users - remove non tracked
		for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
		{
			if (std::find(user_scene_ids.begin(), user_scene_ids.end(), it->first) != user_scene_ids.end())
			{
				// user is in scene - update positional data
			}else
			{
				// user has left scene - delete tracking instance
				delete(it->second);
				// remove mapping
				mFrameIDToUser.erase(it);
			}
		}

		// add new users
		for(int i=0; i<user_scene_ids.size();i++)
		{
			int scene_id = user_scene_ids[i];
			// user not tracked yet - initiate new user model
			if(mFrameIDToUser.count(scene_id) == 0)
			{
				mFrameIDToUser[scene_id] = new User();
			}
		}
	}


	// incorporate processed requests: update user ids
	void ApplyUserIdentification()
	{
		// handle processed requests


		// apply to users
	}

	// send identification requests for all unknown users
	void RequestUserIdentification()
	{
		for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
		{
			if (it->second->GetIDStatus() == IDStatus_Unknown)
			{
				// TODO: make identification request

				// send request id
				//pServerConn->SendChar();


				// send payload
				it->second->SetIDStatus(IDStatus_Pending);
			}
		}
	}

	// ----------------- helper functions

	void DrawUsers(cv::Mat &img)
	{
		for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
		{
			cv::Rect bb = it->second->GetFaceBoundingBox();

			// draw face bounding box
			cv::rectangle(img, bb, cv::Scalar(0, 0, 255), 2, cv::LINE_4);

			// draw identification status
			float font_size = 0.5;
			std::string text;
			enum IdentificationStatus status = it->second->GetIDStatus();
			if(status == IDStatus_Identified)
			{
				text = "Status: identified";
			}else if(status == IDStatus_Pending)
			{
				text = "Status: pending";
			}else
			{
				text = "Status: unknown";
			}
			cv::putText(img, text, cv::Point(bb.x, bb.y), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0, 0, 0), 1, 8);
		}
	}


private:
	io::TCPClient* pServerConn;
	std::map<int, User*> mFrameIDToUser;

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
	if(!c.Connect("127.0.0.1", FLAGS_port))
	{
		std::cout << "Could not connect to server" << std::endl;
		return -1;
	}

	// send request ID
	if(FLAGS_send_batch)
	{
		c.SendChar(3);
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
				// send image batch
				if (FLAGS_send_batch)
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
						if (nr_images == FLAGS_batch_size) {
							// stop recording
							std::cout << "--- Captured " << nr_images << " images" << std::endl;
							int user_id = inputUserID();
							// send batch
							sendTrainingBatch(&c, user_id, image_batch);
							// reset batch
							nr_images = 0;
							image_batch.clear();
						}
					}
				}



			}
		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	// close camera
	k.Close();





	//// receive image
	//cv::Mat server_img = cv::Mat::zeros(96, 96, CV_8UC3);
	//c.ReceiveRGBImage(server_img, 96);
	//// display image
	//cv::imshow("Received from server", server_img);
	//cv::waitKey(0);

	return 0;
}
