#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <opencv2/imgproc/imgproc.hpp>


DEFINE_string(input, "", "Input image folder");
DEFINE_string(output, "converted", "Output folder");
DEFINE_bool(grayscale, false, "Convert images to grayscale");
DEFINE_bool(grayscale_rgb, false, "Convert images to grayscale - save as rgb image");
DEFINE_bool(hsv, false, "Convert images to grayscale");

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if(FLAGS_input == "")
	{
		std::cout << "Please enter an input folder\n";
		return -1;
	}

	// load images0
	io::ImageHandler ih;
	ih.ChangeDirectory(FLAGS_input);

	std::vector<cv::Mat> images;
	std::vector<std::string> filenames;

	while(ih.LoadImageBatch(images, filenames, 10)>0)
	{
		for (size_t i = 0; i<images.size(); i++)
		{
			// process files
			if (FLAGS_grayscale)
			{
				cv::Mat converted;
				cv::cvtColor(images[i], converted, CV_RGB2GRAY);
				ih.SaveImageIndexed(converted, FLAGS_output, filenames[i]);
			}
			else if (FLAGS_grayscale_rgb)
			{
				cv::Mat converted;
				cv::cvtColor(images[i], converted, CV_RGB2GRAY);
				cv::cvtColor(converted, converted, CV_GRAY2RGB);
				ih.SaveImageIndexed(converted, FLAGS_output, filenames[i]);
			}
			else if(FLAGS_hsv)
			{
				cv::Mat converted;
				cv::cvtColor(images[i], converted, CV_RGB2HSV);
				ih.SaveImageIndexed(converted, FLAGS_output, filenames[i]);
			}
			else if (false)
			{

			}
		}
	}




	return 0;
}
