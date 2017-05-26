#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <io\CSVHandling.h>
#include <vector>
#include <recognition/BodyParts.h>
#include <features/Skin.h>

using namespace cv;
using namespace std;


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

	recognition::BodyPartSegmentation bs;


	bool recording = false;
	while (true) {

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {


			Mat img_color;
			Mat img_depth;
			// get color image
			//k.GetImageCopyBGR(img_color);
			k.GetImageCopyBGRSubtracted(img_color);


			k.GetImageCopyDepth8UThresholded(img_depth);

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

			}

			// draw bounding boxes
			if (bounding_boxes_depth.size() > 0)
			{
				
			}

			//img_color = features::SkinDetector::GetSkin(img_color);
			img_color = features::SkinDetector::SubtractSkin(img_color);

			//

			// create binary mask
			cv::Mat mask = cv::Mat::zeros(img_color.size(), CV_8UC1);

			cv::Vec3b cblack = cv::Vec3b::all(0);

			cv::Mat greyMat;
			cv::cvtColor(img_color, greyMat, CV_BGR2GRAY);


			mask.setTo(255, greyMat != 0);

			int erosion_size = 6;
			
			cv::Mat element_dilate = getStructuringElement(cv::MORPH_DILATE,
				cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
				cv::Point(erosion_size, erosion_size));


			//erode(image, dst, element);  // dilate(image,dst,element);
			dilate(mask, mask, element_dilate);

			erosion_size = 12;
			cv::Mat element_erode = getStructuringElement(cv::MORPH_DILATE,
				cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
				cv::Point(erosion_size, erosion_size));

			erode(mask, mask, element_erode);
			erode(mask, mask, element_erode);
			

			




			//cv::imshow("Color", img_color);


			cv::Mat color_img_bodies = cv::Mat::zeros(img_color.size(), img_color.type());


			cv::Mat pants_color(img_color.size(), img_color.type());
			//mat2.setTo(cv::Scalar(12, 12));




			// -----------------------------------------------



			
			


			// get joints
			std::vector<std::vector<cv::Point2f>> user_joints;
			size_t nr_users = 0;

			nr_users = st.GetJointProjections(
				user_joints, 
				base::JointType_HipLeft + base::JointType_HipRight,
				base::ImageSpace_Color, img_color.cols, img_color.rows
				);
			// look for lowest point
			
			int y_max = 0;
			// take first user
			if(nr_users > 0)
			{
				
				for (size_t i_joint=0;i_joint < user_joints[0].size();i_joint++)
				{
					if(y_max < user_joints[0][i_joint].y)
					{
						y_max = user_joints[0][i_joint].y;
					}
				}

			}




			// -----------------------------------------------

			// fill pants with RGB data
			//img_color(pants).copyTo(color_img_bodies(pants), mask(pants));
			//mask(pants_roi) = 0;

			if(y_max > 0)
			{
				cv::Rect pants_roi = cv::Rect2d(0, y_max, img_color.cols, img_color.rows - y_max);
				cv::Rect top_roi = cv::Rect2d(0, 0, img_color.cols, y_max);


				cv::Mat mask_pants = mask.clone();
				cv::Mat mask_top = mask.clone();

				cv::Mat submask_pants(mask_pants, pants_roi);
				cv::Mat submask_top(mask_top, top_roi);
				submask_pants = 0;
				submask_top = 0;

				// draw pants mask
				
				color_img_bodies.setTo(cv::Scalar(20, 160, 200), mask_pants != 0);

				// draw top mask
				color_img_bodies.setTo(cv::Scalar(0, 0, 240), mask_top != 0);

			}



	
			//color_img_bodies(mask).setTo(cv::Scalar(20,160, 200));

			cv::resize(mask, mask, cv::Size(mask.cols / 3, mask.rows / 3));
			cv::imshow("Maks", mask);

			st.RenderAllBodyJoints(color_img_bodies, base::ImageSpace_Color);
			cv::imshow("bodies", color_img_bodies);

			int key = cv::waitKey(3);

		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}	// end while camera loop



	return 0;
}