#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/cvdef.h>

#include <typeindex>
#include <typeinfo>
#include <map>

#include <base/UserIdentification.h>
#include <opencv2/opencv.hpp>
#include <Math/Math.h>

#include <tracking\FaceTracker.h>

void ScopedCopy(math::Array3D<cv::Mat> &arr, cv::Mat*& ptr)
{
	// create in scope
	cv::Mat * scoped = new cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
	// add
	arr.CopyTo(0, 0, 0, *scoped);
	ptr = scoped;
	// mat runs out of scope
	delete(scoped);
}


void ScopedOperatorCopy(math::Array3D<cv::Mat> &arr, cv::Mat*& ptr)
{
	cv::Mat * scoped = new cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));

	// copy operator
	arr(0, 0, 0) = *scoped;
	ptr = scoped;

	// delete mat
	delete(scoped);
}


void BundleTestArray3D() {

	cv::Mat * m = new cv::Mat(cv::Mat::zeros(20, 10, CV_32F));

	// test 1: copying
	math::Array3D<cv::Mat> arr(5, 4, 1);
	cv::Mat * scope_ptr = nullptr;	// check pointer
	ScopedCopy(arr, scope_ptr);
	cv::imshow("red", arr(0, 0, 0));
	cv::waitKey(0);
	cv::destroyAllWindows();

	// test 2: operator overload
	ScopedOperatorCopy(arr, scope_ptr);
	cv::imshow("blue", arr(0, 0, 0));
	cv::waitKey(0);
	cv::destroyAllWindows();

	// test 3: copy from array
	cv::Mat black = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
	black = arr(0, 0, 0);
	arr.Reset(); // reference free memory
	cv::imshow("blue", black);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void BundleTestFaceGrid() {
	tracking::RadialFaceGrid grid;

	cv::Mat in = cv::Mat(100, 200, CV_8UC3, cv::Scalar(255, 0, 0));
	grid.StoreSnapshot(0, 0, 0, in);

	cv::circle(in, cv::Point(0, 0), 20, cv::Scalar(255, 255, 0), cv::LINE_4);

	// draw
	grid.DisplayFaceGridPitchYaw();

	cv::waitKey(0);
}

int main(int argc, char** argv)
{

	BundleTestFaceGrid();

	return 0;
} 
