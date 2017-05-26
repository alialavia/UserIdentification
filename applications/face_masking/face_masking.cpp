#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <imgproc/ImgProcessing.h>

#include <vector>

#include <segmentation/types.h>
#include <segmentation/facemasker.h>

using namespace cv;
using namespace dip;
using namespace std;

const int kWindowWidth = 640;
const int kWindowHeight = 480;

const bool kMasking = true;
const bool kDownsample = true;
const int kMinDepth = 256;
const int kMinPixels = 10;
const int kOpenSize = 2;
const int kHeadWidth = 150;
const int kHeadHeight = 150;
const int kHeadDepth = 100;
const int kFaceSize = 150;
const int kExtendedSize = 50;

const char kCascade[] = "haarcascade_frontalface_default.xml";

void updateColor(cv::Mat src, Color *color) {
	size_t memsize = sizeof(Color) * src.rows * src.cols;
	memcpy(color, src.data, memsize);
}

void updateDepth(cv::Mat src, Depth *depth) {
	size_t memsize = sizeof(Depth) * src.rows * src.cols;
	memcpy(depth, src.data, memsize);
}

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	io::KinectSensorMultiSource k;
	HRESULT hr;


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


	// Initialize face classifier.
	cv::CascadeClassifier cascade;
	if (!cascade.load(kCascade)) {
		printf("Failed to load cascade classifier.\n");
		return -1;
	}

	// Initialize face masker.
	FaceMasker *masker = NULL;
	if (kMasking) {
		masker = new FaceMasker;
		Ptr<cv::BaseCascadeClassifier::MaskGenerator> masker_ptr(masker);
		cascade.setMaskGenerator(masker_ptr);
	}


	cv::Mat color_image;
	cv::Mat depth_image;

	// Initialize buffers.
	Depth *depth = new Depth[512*424];
	Color *color = new Color[1920*1080];

	while (true)
	{

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyBGR(color_image);
			k.GetImageCopyDepth(depth_image);	// depth in mm


			updateColor(color_image, color);
			updateDepth(depth_image, depth);

			Size window_size = cascade.getOriginalWindowSize();

			// Sensor intrinsics
			// https://threeconstants.wordpress.com/2014/11/09/kinect-v2-depth-camera-calibration/
			//Focal Length(x, y) : 391.096, 463.098
			//Principle Point(x, y) : 243.892, 208.922

			// TODO: debug why generateMask fails...
			masker->Run(kMinDepth, kMinPixels, kOpenSize, kHeadWidth, kHeadHeight,
				kHeadDepth, kFaceSize, kExtendedSize, window_size.width,
				512, 424,
				(391.096 + 463.098) / 2.0f,
				depth, color);

			std::vector<cv::Rect> faces;
			cascade.detectMultiScale(color_image, faces);

			// render bounding boxes
			for (size_t i = 0; i < faces.size(); i++) {
				cv::rectangle(color_image, faces[i], cv::Scalar(0, 0, 255));
			}

			// display image
			cv::imshow("Scene", color_image);
			char c = cv::waitKey(3);


		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}

	// close camera
	k.Close();

	return 0;
}
