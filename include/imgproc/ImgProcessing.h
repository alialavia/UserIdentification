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

		// 'TENG' algorithm (Krotkov86)
		static double TENG(const cv::Mat& src, int ksize);

		// 'GLVN' - Graylevel local variance algorithm (Santos97)
		static double GLVN(const cv::Mat& src);
	};

} // namespace

#endif