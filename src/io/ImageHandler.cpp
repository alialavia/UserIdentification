#include <stdio.h>
#include "io/ImageHandler.h"
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>
#include <iostream>
#include <windows.h>
#include <opencv2/video.hpp>

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

// ------------------------ drawing functions

void ImageHandler::PutCenteredVerticalText(std::string text, double font_size, cv::Point pos, cv::Mat& src) {
	// Create mat for text
	cv::Size text_size;
	text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_size, 1, 0);
	cv::Mat textImg = cv::Mat(text_size.height + 1, text_size.width, src.type(), cv::Scalar(255, 255, 255));
	cv::putText(textImg, text, cv::Point(0, text_size.height), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0, 0, 0), 1, 8);

	// rotate
	cv::Mat textImgRotated;
	cv::transpose(textImg, textImgRotated);
	cv::flip(textImgRotated, textImgRotated, 0);

	textImgRotated.copyTo(src(cv::Rect(pos.x - text_size.height / 2, pos.y - text_size.width / 2 - 1, textImgRotated.cols, textImgRotated.rows)));
}

void ImageHandler::DrawCenteredText(std::string text, float font_size, cv::Point pos, cv::Mat &img, cv::Scalar color) {
	cv::Size text_size;
	text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_size, 1, 0);
	cv::putText(img, text, cv::Point(pos.x - text_size.width / 2, pos.y + text_size.height / 2), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1, 8);
}

void ImageHandler::DrawCenteredText(double text, float font_size, cv::Point pos, cv::Mat &img, cv::Scalar color) {
	char str[200];
	sprintf(str, "%.1f", text);
	DrawCenteredText(str, font_size, pos, img, color);
}

