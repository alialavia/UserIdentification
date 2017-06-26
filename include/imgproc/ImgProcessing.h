#ifndef IMGPROC_IMGPROCESSING_H_
#define IMGPROC_IMGPROCESSING_H_

#include <opencv2/core/mat.hpp>
#include <opencv2\imgproc.hpp>

namespace cv {
	class Mat;
}

namespace imgproc {

	class FocusMeasure
	{
	public:
		// 'LAPV' - Variance of Laplacian algorithm (Pech2000)
		static double LAPV(const cv::Mat& src);

		static double LAPD(const cv::Mat& src);

		// 'TENG' algorithm (Krotkov86)
		static double TENG(const cv::Mat& src, int ksize);

		// 'GLVN' - Graylevel local variance algorithm (Santos97)
		static double GLVN(const cv::Mat& src);

		// Maximum of Laplacian - Range: 0-255
		static double MLAP(const cv::Mat& src);

		// Canny Edge count - Range: 0-1
		static double CEC(const cv::Mat& src);
	};

	class ImageProc
	{
	public:
		ImageProc()
		{
			
		}
		static void batchResize(std::vector<cv::Mat> &src, int w, int h) {
			for (size_t i = 0; i < src.size(); i++) {
				cv::resize(src[i], src[i], cv::Size(w, h));
			}
		}

		static cv::Mat createOne(const std::vector<cv::Mat*> & images, int cols, int min_gap_size)
		{
			std::vector<cv::Mat> tmp;

			for (int i = 0; i < images.size(); i++) {
				tmp.push_back(*images[i]);
			}
			return createOne(tmp, cols, min_gap_size);
		}

		static cv::Mat createOne(const std::vector<cv::Mat> & images, int cols, int min_gap_size)
		{
			// find max dimension
			int max_width = 0;
			int max_height = 0;
			for (int i = 0; i < images.size(); i++) {
				// check if type is correct 
				if (i > 0 && images[i].type() != images[i - 1].type()) {
					std::cerr << "WARNING:createOne failed, different types of images";
					return cv::Mat();
				}
				max_height = std::max(max_height, images[i].rows);
				max_width = std::max(max_width, images[i].cols);
			}
			// number of images in y direction
			int rows = std::ceil((float)images.size() / cols);

			// result matrix
			cv::Mat result = cv::Mat::zeros(rows*max_height + (rows - 1)*min_gap_size,
				cols*max_width + (cols - 1)*min_gap_size, images[0].type());
			size_t i = 0;
			int current_height = 0;
			int current_width = 0;
			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {
					if (i >= images.size())
						return result;
					// get the ROI in our result-image
					cv::Mat to(result,
						cv::Range(current_height, current_height + images[i].rows),
						cv::Range(current_width, current_width + images[i].cols));
					// copy the current image to the ROI
					images[i++].copyTo(to);
					current_width += max_width + min_gap_size;
				}
				// next line - reset width and update height
				current_width = 0;
				current_height += max_height + min_gap_size;
			}
			return result;
		}
	};



} // namespace

#endif
