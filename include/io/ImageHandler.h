#ifndef IO_IMAGEHANDLER_H_
#define IO_IMAGEHANDLER_H_

#include <opencv2/core.hpp>


namespace io
{
	class ImageHandler{
		public:
		ImageHandler();
		~ImageHandler();
		
		// file handling
		static bool SaveImage(cv::Mat img, std::string path, std::string filename);
		static bool SaveImageIndexed(cv::Mat img, std::string path, std::string filename);
		static bool FileExists(const std::string& name);
		static void MakeIndexedName(std::string path, std::string &filename_orig, bool start_with_index = true);

		// drawing function
		static void PutCenteredVerticalText(std::string text, double font_size, cv::Point pos, cv::Mat& src);
		static void DrawCenteredText(std::string text, float font_size, cv::Point pos, cv::Mat &img, cv::Scalar color = cv::Scalar(0, 0, 0));
		static void DrawCenteredText(double text, float font_size, cv::Point pos, cv::Mat &img, cv::Scalar color = cv::Scalar(0, 0, 0));
	};
}

#endif