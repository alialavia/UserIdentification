#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <windows.h>


DEFINE_string(input, "", "Input image folder");
DEFINE_string(output, "converted", "Output folder");
DEFINE_bool(grayscale, false, "Convert images to grayscale");
DEFINE_bool(hsv, false, "Convert images to grayscale");

bool IsImage(const std::string &filename) {
	std::string ext = filename.substr(filename.find_last_of(".") + 1);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	if (
		ext == "jpg" ||
		ext == "jpeg" ||
		ext == "png" ||
		ext == "gif"
		) {
		return true;
	}
	else {
		return false;
	}
}

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (FLAGS_input == "")
	{
		std::cout << "Please enter an input folder\n";
		return -1;
	}

	// load images0
	io::ImageHandler ih;
	ih.ChangeDirectory(FLAGS_input);

	std::vector<cv::Mat> images;
	std::vector<std::string> filenames;

	std::string mDirectory = "";
	WIN32_FIND_DATA mCurrentFile;
	HANDLE mDirHandle = INVALID_HANDLE_VALUE;


	//----------------
	
	// check if exists
	if ((mDirHandle = FindFirstFile((FLAGS_input + "/*").c_str(), &mCurrentFile)) == INVALID_HANDLE_VALUE) {
		// no files found
		std::cout << "No files" << std::endl;
		return -1;
	}

	int batch_size = 10;
	int i = 0;
	cv::Mat image;

	bool all_files = false;
	do {

		const bool is_directory = (mCurrentFile.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
		if (is_directory)
			continue;

		const std::string file_name = mCurrentFile.cFileName;
		const std::string full_file_name = FLAGS_input + "/" + file_name;

		if (file_name[0] == '.')
			continue;

		if (!IsImage(file_name))
			continue;

		// load image
		image = cv::imread(full_file_name, CV_LOAD_IMAGE_COLOR);   // Read the file
		if (!image.data) {
			// could not open file
		}
		else {
			images.push_back(image);
			filenames.push_back(file_name);
		}

		// increment batch size
		i++;

		std::cout << file_name << std::endl;
	} while (i < batch_size && (all_files = FindNextFile(mDirHandle, &mCurrentFile)) != 0);


	std::cout << all_files << std::endl;

	FindNextFile(mDirHandle, &mCurrentFile);
	const std::string file_name = mCurrentFile.cFileName;
	std::cout << file_name << std::endl;



	return 0;
	//----------------------------------

	return 0;
}
