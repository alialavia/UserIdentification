#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>

int main(int argc, char** argv)
{
	io::KinectSensorMultiSource k;
	HRESULT hr;

	cv::Mat color_image;

	// initialize sensor
	if (FAILED(k.Open())) {
		std::cout << "Initialization failed" << std::endl;
		return -1;
	}

	// get sensor reference
	IKinectSensor* pSensor = nullptr;

	if (FAILED(k.GetSensorReference(pSensor)))
	{
		std::cout << "Sensor is not initialized" << std::endl;
		return -1;
	}


	cvNamedWindow("Color image", CV_WINDOW_AUTOSIZE);

	while (true) {

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			k.GetImageCopyRGB(color_image);
			cv::imshow("Color image", color_image);
			cv::waitKey(3);

		} else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}


	return 0;
}
