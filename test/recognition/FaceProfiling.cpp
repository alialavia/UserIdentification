#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>


#include "tracking/SkeletonTracker.h"

int main(int argc, char** argv)
{
	io::KinectSensorMultiSource k;


	HRESULT hr;
	cvNamedWindow("Color image", CV_WINDOW_AUTOSIZE);

	cv::Mat color_image;

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
	st.Init();

	while (true) {

		// polling - buffers refreshed
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// extract skeleton data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			// draw bounding boxes
			//st.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);



			// get face bounding boxes
			std::vector<cv::Rect2f> bounding_boxes;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color, color_image.cols, color_image.rows);

			if(bounding_boxes.size() > 0)
			{
				cv::Mat face = color_image(bounding_boxes[0]);
				// show image
				cv::imshow("Color image", face);
				cv::waitKey(3);
			}



		} else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}


	return 0;
}
