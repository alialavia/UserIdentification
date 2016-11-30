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

} // namespace

#endif
