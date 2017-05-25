#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>


#include <user\BatchUserManager.h>
#include <user\StreamUserManager.h>
#include <user\User.h>
#include <io/Networking.h>
#include <io/RequestHandler.h>
#include <imgproc/ImgProcessing.h>


DEFINE_int32(port, 8080, "Server port");
DEFINE_bool(face_dect, false, "Detect faces parallel to Kinect.");

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

#ifdef _KEEP_SERVER_CONNECTION
	server_conn.Connect();
#endif
	

	// start request handler
	io::NetworkRequestHandler req_handler;
	req_handler.start(); // parallel processing

	// user manager
	//user::BatchUserManager um;
	user::StreamUserManager um;
	if (!um.Init(&server_conn, &req_handler)) {
		std::cout << "Could not initialize user manager" << std::endl;
		return -1;
	}

	// threaded dlib face detector
	features::AsyncFaceDetector face_dect;
	if (FLAGS_face_dect) {
		face_dect.start();
	}

	// print instructions
	std::cout << "=====================================\n"
				"          INSTRUCTIONS\n"
				"=====================================\n"
				"[1]: Show all profile pictures of the server\n"
				"[d]: Render debug information\n"
				"[q]: Disconnect client\n"
				"--------------------------------------\n"
				"\n\n";

	while (true) 
	{

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// async parallel face detection
			if (FLAGS_face_dect) {
				face_dect.TryToDetectFaces(color_image);
				//std::cout << "==== Faces: " << face_dect.GetNrFaces() << std::endl;
			}

			// extract skeleton data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			std::vector<cv::Rect2f> bounding_boxes;
			std::vector<cv::Point3f> user_positions;
			std::vector<int> user_scene_ids;

			// extract face bb from skeleton: extract corresponding users ids
			// this is not the same as st.GetUserSceneIDs(user_scene_ids); - this gives all users in the scene (even not trackeable users)
			st.GetFaceBoundingBoxesRobust(bounding_boxes, user_scene_ids, base::ImageSpace_Color);

			st.GetJointPosition(base::JointType_SpineMid, user_positions);

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
			//if (user_scene_ids.size() > 0)
			{
				// refresh users (add/remove users, reset features)
				um.RefreshUserTracking(user_scene_ids, bounding_boxes, user_positions);

				// ------------- update features
				// face data
				um.UpdateFaceData(faces, face_ids);

				// skeleton data

				// color model
				// ...

				// ------------- 
				// update the tracking status (safety/human tracking) based on features
				um.UpdateTrackingStatus();

				// ------------- 

				// Process responses
				// - update user ids
				um.ProcessResponses();

				// Generate requests
				// - request identification for unknown users
				// - update classifiers for known users
				um.GenerateRequests(color_image);

				// draw users
				um.RenderGUI(color_image);
			}

			// display image
			cv::imshow("Scene", color_image);
			char c = cv::waitKey(3);

			// get all profile pictures from server
			if (c == '1')
			{
				//::vector<std::pair<int, cv::Mat>> profile_pictures = um.GetAllProfilePictures();
				std::vector<int> user_ids;
				std::vector<cv::Mat> profile_pics;

				// get profile pictures
				if (!um.GetAllProfilePictures(profile_pics, user_ids)) {
					std::cout << "Could not acquire profile pictures...\n";
					continue;
				}

				// write id on profile pictures
				for(size_t i=0;i<profile_pics.size();i++)
				{
					cv::Rect bg_patch = cv::Rect(0, 0, 30, 20);
					profile_pics[i](bg_patch) = um.GetUserColor(user_ids[i]);
					cv::putText(profile_pics[i], "ID"+std::to_string(user_ids[i]), cv::Point(8, 12), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1, 8);
				}

				if(profile_pics.size() > 0)
				{
					cv::Mat combined = imgproc::ImageProc::createOne(profile_pics, 5, 10);
					cv::imshow("Profile Pictures", combined);
					cv::waitKey(3);
				}else
				{
					std::cout << "No profile pictures taken yet...\n";
				}
			}
			else if(c == 'd') {
				um.ToggleRenderDebug();
			}else if(c == 'q')
			{
				// disconnect client
				break;
			}

		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	if (FLAGS_face_dect) {
		face_dect.stop();
	}

	// close camera
	k.Close();

#ifdef _KEEP_SERVER_CONNECTION
	// send disconnection signal to server
	io::ClientDisconnect disc(&server_conn);
	disc.SubmitRequest();
	server_conn.Close();
#endif


	return 0;
}
