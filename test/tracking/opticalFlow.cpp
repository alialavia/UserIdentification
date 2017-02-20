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
			k.GetImageCopyRGB(img_color);
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


//
//
//
//int main(int argc, char** argv)
//{
//	io::KinectSensorMultiSource k;
//
//
//	HRESULT hr;
//
//	// initialize sens
//	if (FAILED(k.Open())) {
//		std::cout << "Initialization failed" << std::endl;
//		return -1;
//	}
//
//	// skeleton tracker
//	IKinectSensor* pSensor = nullptr;
//
//	if (FAILED(k.GetSensorReference(pSensor)))
//	{
//		std::cout << "Sensor is not initialized" << std::endl;
//		return -1;
//	}
//
//	tracking::SkeletonTracker st(pSensor);
//	st.Init();
//
//
//	cv::Mat face_snap;
//
//
//	Mat flow, frame;
//	// some faster than mat image container
//	UMat  flowUmat, prevgray;
//
//	while (true) {
//
//		// polling
//		hr = k.AcquireFrame();
//
//		// check if there is a new frame available
//		if (SUCCEEDED(hr)) {
//
//
//			Mat img_color;
//			Mat original;
//
//			// get color image
//			k.GetImageCopyRGB(img_color);
//
//			resize(img_color, img_color, Size(640, 480));
//
//			// save original for later
//			img_color.copyTo(original);
//
//			// just make current frame gray
//			cvtColor(img_color, img_color, COLOR_BGR2GRAY);
//
//			if (prevgray.empty() == false) {
//
//				// calculate optical flow 
//				calcOpticalFlowFarneback(prevgray, img_color, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
//				// copy Umat container to standard Mat
//				flowUmat.copyTo(flow);
//
//
//				// By y += 5, x += 5 you can specify the grid 
//				for (int y = 0; y < original.rows; y += 5) {
//					for (int x = 0; x < original.cols; x += 5)
//					{
//						// get the flow from y, x position * 10 for better visibility
//						const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;
//						// draw line at flow direction
//						line(original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 0, 0));
//						// draw initial point
//						circle(original, Point(x, y), 1, Scalar(0, 0, 0), -1);
//
//
//					}
//
//				}
//
//				// draw the results
//				namedWindow("prew", WINDOW_AUTOSIZE);
//				imshow("prew", original);
//
//				// fill previous image again
//				img_color.copyTo(prevgray);
//
//			}
//			else {
//
//				// fill previous image in case prevgray.empty() == true
//				img_color.copyTo(prevgray);
//
//			}
//
//
//			int key1 = waitKey(20);
//		}
//		else {
//			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)
//
//		}
//	}	// end while camera loop
//
//
//
//	return 0;
//}
