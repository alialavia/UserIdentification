#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/FaceTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>


DEFINE_string(img_folder, "", "Input image folder");
DEFINE_string(log_name, "image_labels.csv", "Output log for the image labels");

int main(int argc, char** argv)
{

	// print instructions
	std::cout << "=====================================\n"
		"          INSTRUCTIONS\n"
		"=====================================\n"
		"[any key]: press any key when an images shows to assign it as label\n"
		"[q]: Quit\n"
		"--------------------------------------\n"
		"\n\n";

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	io::ImageHandler ih;
	io::CSVWriter csv(FLAGS_log_name);

	cv::Mat color_image;
	int key = (int)('-1');
	std::vector<cv::Mat> img_batch;
	std::vector<std::string> img_filenames;
	cv::Mat canvas;

	if (ih.ChangeDirectory(FLAGS_img_folder)) {
		while (ih.LoadImageBatch(img_batch, img_filenames, 1) > 0) {
			// display image
			cv::imshow("Image", img_batch[0]);
			key = cv::waitKey(0);

			// quit
			if ((int)('q') == key) {
				break;
			}

			//while (key < 10) {
			//	key = cv::waitKey(0);
			//}

			// save label
			csv.addEntry(img_filenames[0]);
			csv.addEntry(key);
			csv.EndRow();
		}
		std::cout << "All images processed." << std::endl;
	}
	else {
		std::cout << "Could not open folder" << std::endl;
	}

	return 0;
}
