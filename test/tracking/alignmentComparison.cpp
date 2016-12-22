#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/FaceTracker.h"
#include <gflags/gflags.h>

#include <io/Networking.h>
#include <io/RequestTypes.h>
#include <io/ResponseTypes.h>

#include <tracking\SkeletonTracker.h>
#include <features\Face.h>

DEFINE_int32(port, 8080, "Server port");

int main(int argc, char** argv)
{

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// config to server connection
	io::TCPClient server_conn;
	server_conn.Config("127.0.0.1", FLAGS_port);

	HRESULT hr;
	cv::Mat color_image;
	cv::Mat face_snap;

	// initialize sensor
	io::KinectSensorMultiSource k;
	if (FAILED(k.Open())) {
		std::cout << "Initialization failed" << std::endl;
		return -1;
	}

	// extract sensor reference
	IKinectSensor* pSensor = nullptr;
	if (FAILED(k.GetSensorReference(pSensor)))
	{
		std::cout << "Sensor is not initialized" << std::endl;
		return -1;
	}

	// skeleton tracker
	tracking::SkeletonTracker st(pSensor);
	st.Init();

	// dlib aligner
	features::DlibFaceAligner2 dlib_aligner;
	dlib_aligner.Init();

	// kinect aligner
	features::KinectFaceAligner k_aligner(pSensor);

	// face tracker
	tracking::FaceTracker ft(pSensor);
	
	while (true) {

		// polling
		hr = k.AcquireFrame();
		if (SUCCEEDED(hr)) {

		}

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// extract joint data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			// extract raw face data
			FaceData* face_data_raw = k.GetFaceDataReference();
			ft.ExtractFacialData(face_data_raw);

			k_aligner.ExtractFacialData(face_data_raw);

			// get face bounding boxes from face tracking
			//std::vector<cv::Rect2f> bounding_boxes;
			//ft.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);

			// get face bounding boxes from skeleton tracking
			std::vector<cv::Rect2f> bounding_boxes;
			std::vector<int> user_ids;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, user_ids, base::ImageSpace_Color);

			if (bounding_boxes.size() > 0)
			{

				face_snap = color_image(bounding_boxes[0]);

				// request requires quadratic image
				cv::resize(face_snap, face_snap, cv::Size(100, 100));

				// connect to server
				if (server_conn.Connect())
				{

					// generate request
					io::ImageAlignment req(&server_conn, face_snap);
					req.SubmitRequest();

					// get reponse
					io::QuadraticImageResponse response(&server_conn);
					if (!response.Load()) {
						std::cout << "--- An error occurred during alignment\n";
					}
					else {
					
						cv::imshow("Remote aligned face", response.mImage);
						int key = cv::waitKey(3);
						if (key == 32)	// space = quite
						{
							break;
						}
					}

					// close server connection
					server_conn.Close();
				}

			}	// endif faces detected


			// draw bounding boxes
			ft.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);
			ft.RenderFaceFeatures(color_image, base::ImageSpace_Color);

			// show image
			cv::imshow("Faces", color_image);
			int key = cv::waitKey(3);
			if (key == 32)	// space = quite
			{
				break;
			}

		} else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}	// end while camera loop

	return 0;
}
