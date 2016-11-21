#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>


#include <user\UserManager.h>
#include <user\User.h>
#include <io/Networking.h>
#include <io/RequestHandler.h>

DEFINE_int32(port, 8080, "Server port");

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	io::KinectSensorMultiSource k;
	HRESULT hr;
	cv::Mat color_image;

	// initialize sensor
	if (FAILED(k.Open())) {
		std::cout << "Initialization failed" << std::endl;
		return -1;
	}

	// get sensor reference
	IKinectSensor* pSensor = nullptr;

	if (FAILED(k.GetSensorReference(pSensor)))
	{
		std::cout << "Sensor is not initialized" << std::endl;
		return -1;
	}

	// skeleton tracker
	tracking::SkeletonTracker st(pSensor);
	if (FAILED(st.Init()))
	{
		std::cout << "Skeleton tracker initialization failed" << std::endl;
		return -1;
	}

	// connect to server
	io::TCPClient server_conn;
	if(!server_conn.Connect("127.0.0.1", FLAGS_port))
	{
		std::cout << "Could not connect to server - 127.0.0.1:" << FLAGS_port << std::endl;
		return -1;
	}

	// start request handler
	io::NetworkRequestHandler req_handler;
	req_handler.start(); // parallel processing

	// user manager
	user::UserManager um;
	if (!um.Init(&server_conn, &req_handler)) {
		std::cout << "Could not initialize user manager" << std::endl;
		return -1;
	}


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

			// if users in scene
			if (user_scene_ids.size() > 0)
			{
				// refresh users
				um.RefreshTrackedUsers(user_scene_ids, bounding_boxes);

				// update user ids
				um.ApplyUserIdentification();

				// draw users
				um.DrawUsers(color_image);

				// request identification for unknown users
				um.RequestUserIdentification(color_image);
			}

		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	// close camera
	k.Close();



	return 0;
}
