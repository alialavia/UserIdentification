#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include "io/CSVHandling.h"
#include "io/ImageHandler.h"

DEFINE_string(log_name, "tracking_log.csv", "Output log (e.g. tracking_log.csv)");

int main(int argc, char** argv)
{
	io::KinectSensorMultiSource k;


	HRESULT hr;
	cvNamedWindow("Color image", CV_WINDOW_AUTOSIZE);

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

	if(FAILED(k.GetSensorReference(pSensor)))
	{
		std::cout << "Sensor is not initialized" << std::endl;
		return -1;
	}

	tracking::SkeletonTracker st(pSensor);
	st.TrackVelocity(false);
	st.Init();
	bool skeleton_tracked = false;
	io::CSVWriter csv("tracking_lost.csv");
	io::CSVWriter csv2("tracking_starts.csv");
	std::vector<cv::Point3f> user_joints;
	io::ImageHandler ih;

	while (true) {

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyBGR(color_image);

			// extract skeleton data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			// get face bounding boxes
			std::vector<cv::Rect2f> bounding_boxes;
			std::vector<int> user_ids;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, user_ids, base::ImageSpace_Color);

			if (bounding_boxes.size() > 0)
			{
				
				st.GetJointPosition(base::JointType_SpineMid, user_joints);
				if(!skeleton_tracked)
				{
					for (size_t i = 0; i<user_joints.size(); i++)
					{
						if (user_joints[i].z>0.07)
						{
							csv2.addEntry(user_joints[i].x);
							csv2.addEntry(user_joints[i].y);
							csv2.addEntry(user_joints[i].z);
							csv2.EndRow();
							// take picture
							ih.SaveImageIndexed(color_image, "", "tracking_starts.png");
							std::cout << "--- tracking started at: " << user_joints[i].x << "/" << user_joints[i].y << "/" << user_joints[i].z << std::endl;
						}
					}
				}
				skeleton_tracked = true;
			}else
			{
				if(skeleton_tracked)
				{
					skeleton_tracked = false;
					// log distance
					for(size_t i = 0;i<user_joints.size();i++)
					{
						if(user_joints[i].z>0.07)
						{
							csv.addEntry(user_joints[i].x);
							csv.addEntry(user_joints[i].y);
							csv.addEntry(user_joints[i].z);
							csv.EndRow();
							// take picture
							ih.SaveImageIndexed(color_image, "", "tracking_ends.png");
							std::cout << "--- tracking lost at: " << user_joints[i].x << "/" << user_joints[i].y << "/" << user_joints[i].z << std::endl;
						}
					}
				}
			}

			// draw bounding boxes
			st.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);

			// show image
			cv::imshow("Color image", color_image);
			cv::waitKey(3);

		} else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}


	return 0;
}
