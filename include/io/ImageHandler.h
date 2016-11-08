#ifndef IO_IMAGEHANDLER_H_
#define IO_IMAGEHANDLER_H_

#include <opencv2/core.hpp>


namespace io
{
	class ImageHandler{
		public:
		ImageHandler();
		~ImageHandler();
		
		static bool SaveImage(cv::Mat img, std::string path, std::string filename);
		static bool SaveImageIndexed(cv::Mat img, std::string path, std::string filename);
		static bool FileExists(const std::string& name);
		static void MakeIndexedName(std::string path, std::string &filename_orig, bool start_with_index = true);

	};
}

#endif