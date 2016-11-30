#include <iostream>
#include <imgproc\ImgProcessing.h>

#include <opencv2\highgui.hpp>
int main(int argc, char** argv)
{
	
	cv::Mat img = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
	imgproc::FocusMeasure::LAPD(img);


	cv::imshow("basdf", img);
	cv::waitKey(0);


	return 0;
} 
