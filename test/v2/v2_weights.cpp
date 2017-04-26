#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/FaceTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>

#include <tracking\SkeletonTracker.h>
#include <features\Face.h>


int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);


	io::KinectSensorMultiSource k;


	HRESULT hr;

	cv::Mat color_image;

	// initialize sens
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

	tracking::SkeletonTracker st(pSensor);
	st.Init();

	// dlib aligner
	//features::DlibFaceAligner dlib_aligner;
	//dlib_aligner.Init();

	tracking::FaceTracker ft(pSensor);
	cv::Mat face_snap;


	while (true) {

		// polling
		hr = k.AcquireFrame();
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// extract joint data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			// extract raw face data
			FaceData* face_data_raw = k.GetFaceDataReference();
			ft.ExtractFacialData(face_data_raw);

			// get face bounding boxes from skeleton tracking
			std::vector<cv::Rect2f> bounding_boxes;
			std::vector<int> user_ids;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, user_ids, base::ImageSpace_Color);



			// faces
			std::vector<tracking::Face> faces;
			ft.GetFaces(faces);

			if (bounding_boxes.size() > 0 && faces.size() > 0 && bounding_boxes.size() == faces.size())
			{

				for (size_t i = 0; i < bounding_boxes.size();i++) {
					face_snap = color_image(bounding_boxes[i]).clone();

					int roll, pitch, yaw;
					float font_size = 0.5;

					faces[i].IsFrontal(true);
					faces[i].GetEulerAngles(roll, pitch, yaw);

					// visualize head pose
					cv::Point center = cv::Point(face_snap.rows / 2, face_snap.cols / 2);
					float max_yaw = 40.;
					float max_pitch = 40.;
					float max_length = 100.;
					cv::Point p0 = center;
					cv::Point p1 = center - cv::Point(yaw/max_yaw*max_length, pitch/max_pitch*max_length);

					cv::arrowedLine(face_snap, p0, p1, cv::Scalar(0,0,255), 2);

					cv::putText(face_snap, "Weight: "+std::to_string(tracking::RadialFaceGrid::CalcSampleWeight(roll, pitch, yaw)), cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(255,255,255), 1, 8);
					cv::putText(face_snap, "P: "+std::to_string(pitch)+" | Y: "+std::to_string(yaw), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(255,255,255), 1, 8);
					cv::imshow("test", face_snap);
					cv::waitKey(2);
				}

			}
			else {
				cv::imshow("test", color_image);
				cv::waitKey(2);
			}

		}


	}


	return 0;
}
