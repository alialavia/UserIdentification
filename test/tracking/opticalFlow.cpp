#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <io\CSVHandling.h>
#include <vector>
#include <tracking\OpticalFlow.h>


using namespace cv;
using namespace std;

DEFINE_bool(depth, false, "Calculate optical flow from depth");

int main(int argc, char** argv)
{

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	io::KinectSensorMultiSource k;

	HRESULT hr;

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


	std::cout <<
		"=============================\n"
		"Press [space] to start logging the optical flow\n"
		"Press [space] again to quit the application and save the log\n"
		"=============================\n";

	tracking::SkeletonTracker st(pSensor);
	st.Init();

	tracking::OpticalFlow of;
	std::vector<double> flog_log;

	bool recording = false;
	while (true) {

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {


			Mat img_color;
			Mat img_depth;
			// get color image
			k.GetImageCopyBGR(img_color);
			k.GetImageCopyDepth8UThresholded(img_depth);
			
			//cv::cvtColor(img_depth, img_depth, CV_GRAY2RGB);

			// extract joint data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			// get face bounding boxes from skeleton tracking
			std::vector<cv::Rect2f> bounding_boxes;
			std::vector<cv::Rect2f> bounding_boxes_depth;
			std::vector<int> user_ids;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, user_ids, base::ImageSpace_Color);
			st.GetFaceBoundingBoxesRobust(bounding_boxes_depth, user_ids, base::ImageSpace_Depth);

			// draw bounding boxes
			if (bounding_boxes.size() > 0)
			{
				for (size_t i = 0; i < bounding_boxes.size();i++) {
					cv::rectangle(img_color, bounding_boxes[i], cv::Scalar(0, 14, 88), 2, cv::LINE_4);
				}

				if (!FLAGS_depth) {
					of.UpdateFlow(img_color(bounding_boxes[0]));
					double flow = of.GetAbsoluteFlow();
					if (recording) {
						flog_log.push_back(flow);
					}
					std::cout << "Flow: " << flow << std::endl;
					of.DispFlow();
				}
			}

			// draw bounding boxes
			if (bounding_boxes_depth.size() > 0)
			{
				for (size_t i = 0; i < bounding_boxes_depth.size(); i++) {
					cv::rectangle(img_depth, bounding_boxes_depth[i], cv::Scalar(0, 14, 88), 2, cv::LINE_4);
				}

				//of.UpdateFlow(img_color(bounding_boxes[0]));
				if (FLAGS_depth) {
					of.UpdateFlow(img_depth(bounding_boxes_depth[0]));
					double flow = of.GetAbsoluteFlow();
					if (recording) {
						flog_log.push_back(flow);
					}
					std::cout << "Flow: " << flow << std::endl;
					of.DispFlow();
				}

			}

			if (!FLAGS_depth) {
				cv::imshow("Depth", img_color);
			}
			else {
				cv::imshow("Depth", img_depth);
			}
			
			int key = cv::waitKey(3);
			if (key == 32)	// space = save
			{
				if (recording==false) {
					recording = true;
					std::cout << "Start logging...\n";
				}
				else {
					break;
				}
				
			}
		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}	// end while camera loop

	// save logs
	if (flog_log.size()>0) {
		std::cout << "Saving log...\n";
		io::CSVWriter out("optical_flow_log.csv");
		out.addList(flog_log);
	}


	return 0;
}
