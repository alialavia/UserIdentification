#include <stdio.h>
#include "io/ImageHandler.h"
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>
#include <iostream>
#include <windows.h>

using namespace io;

ImageHandler::ImageHandler()
{

}

ImageHandler::~ImageHandler()
{
	
}

void ImageHandler::MakeIndexedName(std::string path, std::string &filename_orig, bool start_with_index)
{

	size_t lastindex = filename_orig.find_last_of(".");
	std::string rawname = filename_orig;
	std::string extension = "";

	// separate extension
	if (lastindex != std::string::npos) {
		rawname = filename_orig.substr(0, lastindex);
		extension = filename_orig.substr(lastindex + 1);
	}

	int index = 0;
	std::string filename_tmp = filename_orig;

	if(start_with_index)
	{
		filename_tmp = rawname + "_" + std::to_string(index);
		if (lastindex != std::string::npos)
		{
			filename_tmp += "." + extension;
		}
	}

	// generate unique name
	while (FileExists(path + filename_tmp))
	{
		index++;
		filename_tmp = rawname + "_" + std::to_string(index);

		if (lastindex != std::string::npos)
		{
			filename_tmp += "." + extension;
		}
	}

	if(index > 0 || start_with_index)
	{
		filename_orig = rawname + "_" + std::to_string(index);
		if (lastindex != std::string::npos)
		{
			filename_orig += "." + extension;
		}
	}

}

bool ImageHandler::SaveImageIndexed(cv::Mat img, std::string path, std::string filename) {

	if (CreateDirectory(path.c_str(), NULL))
	{
		// Directory created
	}

	// unique filename
	MakeIndexedName(path, filename);
	cv::imwrite(path + filename, img);
	return true;
}

bool ImageHandler::SaveImage(cv::Mat img, std::string path, std::string filename){

	if (CreateDirectory(path.c_str(), NULL))
	{
		// Directory created
	}

	// unique filename
	MakeIndexedName(path, filename, false);
	cv::imwrite(path + filename, img);
	return true;
}

bool ImageHandler::FileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}