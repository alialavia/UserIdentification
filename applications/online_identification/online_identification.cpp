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

	// face tracker
	tracking::FaceTracker ft(pSensor);

	// config to server connection
	io::TCPClient server_conn;
	server_conn.Config("127.0.0.1", FLAGS_port);

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

			std::vector<cv::Rect2f> bounding_boxes;
			std::vector<int> user_scene_ids;

			// extract face bb from skeleton: extract corresponding users ids
			// this is not the same as st.GetUserSceneIDs(user_scene_ids); - this gives all users in the scene (even not trackeable users)
			st.GetFaceBoundingBoxesRobust(bounding_boxes, user_scene_ids, base::ImageSpace_Color);

			// extract raw face data
			FaceData* face_data_raw = k.GetFaceDataReference();
			ft.ExtractFacialData(face_data_raw);

			// get faces
			std::vector<tracking::Face> faces;
			std::vector<int> face_ids;
			ft.GetFaces(faces, face_ids);

			// extract face bb from face tracker - more unstable
			// else, face data might not be available all the time
			//ft.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);
			//ft.GetUserSceneIDs(user_scene_ids);

			// if users in scene
			if (user_scene_ids.size() > 0)
			{
				// refresh users (add/remove users, reset features)
				um.RefreshUserTracking(user_scene_ids, bounding_boxes);

				// ------------- update features
				// face data
				um.UpdateFaceData(faces, face_ids);
				// skeleton data

				// color model
				// ...

				// ------------- 

				// Process responses
				// - update user ids
				um.ProcessResponses();

				// Generate requests
				// - request identification for unknown users
				// - update classifiers for known users
				um.GenerateRequests(color_image);

				// draw users
				um.DrawUsers(color_image);
			}

			// display image
			cv::imshow("Scene", color_image);
			cv::waitKey(3);

		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	// close camera
	k.Close();



	return 0;
}
