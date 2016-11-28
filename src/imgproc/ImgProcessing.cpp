#include <math/Math.h>

#include <imgproc\ImgProcessing.h>

using namespace imgproc;

double FocusMeasure::LAPV(const cv::Mat& src)
{
	cv::Mat lap;
	cv::Laplacian(src, lap, CV_64F);
	cv::Scalar mu, sigma;
	cv::meanStdDev(lap, mu, sigma);
	double focusMeasure = sigma.val[0] * sigma.val[0];
	return focusMeasure;
}

double FocusMeasure::TENG(const cv::Mat& src, int ksize)
{
	cv::Mat Gx, Gy;
	cv::Sobel(src, Gx, CV_64F, 1, 0, ksize);
	cv::Sobel(src, Gy, CV_64F, 0, 1, ksize);
	cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);
	double focusMeasure = cv::mean(FM).val[0];
	return focusMeasure;
}

double FocusMeasure::GLVN(const cv::Mat& src) {
	cv::Scalar mu, sigma;
	cv::meanStdDev(src, mu, sigma);
	double focusMeasure = (sigma.val[0] * sigma.val[0]) / mu.val[0];
	return focusMeasure;
}