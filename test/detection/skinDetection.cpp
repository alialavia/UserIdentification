#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <features/Skin.h>

DEFINE_string(output, "output", "Output path");

int main(int argc, char** argv)
{
	io::KinectSensorMultiSource k;


	HRESULT hr;
	cvNamedWindow("Color image", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Face", CV_WINDOW_AUTOSIZE);

	cv::Mat color_image;
	INT64 timestamp = 0;
	INT64 timestamp_old = 0;
	INT64 dT = 0;

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

	tracking::SkeletonTracker st(pSensor);
	st.TrackVelocity(true);
	st.Init();

	while (true) {

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyBGR(color_image);

			// extract skeleton data
			IBody** bodies = k.GetBodyDataReference();
			timestamp_old = timestamp;
			timestamp = k.GetBodyTimeStamp();
			dT = timestamp - timestamp_old;

			//std::cout << timestamp << std::endl;
			//std::cout << "dT:" << dT/10000. << " ms"<< std::endl;
			st.ExtractJoints(bodies, timestamp);

			// get face bounding boxes
			std::vector<cv::Rect2f> bounding_boxes;
			std::vector<int> user_ids;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, user_ids, base::ImageSpace_Color);

			if (bounding_boxes.size() > 0)
			{
				cv::Mat face = color_image(bounding_boxes[0]);


				cv::Mat skin = features::SkinDetector::GetSkin(face);

				// show image
				cv::imshow("Skin", skin);
				int key = cv::waitKey(3);




			}

			// draw bounding boxes
			st.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);

			// show image
			cv::imshow("Color image", color_image);
			cv::waitKey(3);

		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}


	return 0;
}
