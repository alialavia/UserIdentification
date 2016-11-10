#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/SkeletonTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>

DEFINE_string(output, "output", "Output path");

int main(int argc, char** argv)
{
	io::KinectSensorMultiSource k;


	HRESULT hr;
	cvNamedWindow("Color image", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Face", CV_WINDOW_AUTOSIZE);

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

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// extract skeleton data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			// get face bounding boxes
			std::vector<cv::Rect2f> bounding_boxes;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);

			if (bounding_boxes.size() > 0)
			{
				cv::Mat face = color_image(bounding_boxes[0]);
				// show image
				cv::imshow("Face", face);
				int key = cv::waitKey(3);
				if (key == 32)	// space = save
				{
					std::string filename = "face.png";
					std::string path = FLAGS_output;

					// save image
					io::ImageHandler::SaveImage(face, path, filename);
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
