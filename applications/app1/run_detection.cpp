#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>


void testChange(cv::Mat &dst) {

	cv::Mat test = cv::Mat::zeros(600, 800, CV_8UC4);
	dst = test;
}

	//testChange(color_image);
	//cv::imshow("Color image", color_image);
	//cv::waitKey(3000);

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


	while (true) {

		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			k.GetColorImageCopy(color_image);

			// crash much?
			cv::imshow("Color image", color_image);
			cv::waitKey(3);

		} else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}


	return 0;
}
