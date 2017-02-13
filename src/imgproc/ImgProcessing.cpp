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

double FocusMeasure::LAPD(const cv::Mat& src)
{
	cv::Mat k1 = cv::Mat::zeros(1, 3, CV_32F);
	k1.at<float>(0) = -1;
	k1.at<float>(1) = 2;
	k1.at<float>(2) = -1;
	cv::Mat k2 = cv::Mat::zeros(3, 3, CV_32F)/(float)sqrt(2);
	k2.at<float>(0,2) = -1;
	k2.at<float>(1,1) = 2;
	k2.at<float>(2,0) = -1;
	cv::Mat k3 = cv::Mat::zeros(3, 3, CV_32F)/(float)sqrt(2);

	cv::Mat dst1, dst2, dst3;
	cv::Point anchor;
	double delta;
	int ddepth;
	anchor = cv::Point(-1, -1);
	delta = 0;
	ddepth = -1;

	// convolve
	cv::filter2D(src, dst1, ddepth, k1, anchor, delta, cv::BORDER_DEFAULT);
	cv::filter2D(src, dst2, ddepth, k2, anchor, delta, cv::BORDER_DEFAULT);
	cv::filter2D(src, dst3, ddepth, k3, anchor, delta, cv::BORDER_DEFAULT);
	cv::Mat absSum = cv::abs(dst1) + cv::abs(dst2) + cv::abs(dst3) + cv::abs(dst1);
	cv::Scalar tempVal = cv::mean(absSum);
	return tempVal.val[0];
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

double FocusMeasure::MLAP(const cv::Mat& src) {
	cv::Mat lap;
	cv::Laplacian(src, lap, CV_64F);
	cv::convertScaleAbs(lap, lap);
	double min, max;
	cv::minMaxLoc(lap, &min, &max);
	return max;
}

double FocusMeasure::CEC(const cv::Mat& src) {

	unsigned long int sum = 0;
	unsigned long int size = src.cols * src.rows;
	cv::Mat edges = src.clone();
	GaussianBlur(edges, edges, cv::Size(7, 7), 1.5, 1.5);
	cv::Canny(edges, edges, 0, 30, 3);

	cv::MatIterator_<uchar> it, end;
	for (it = edges.begin<uchar>(), end = edges.end<uchar>(); it != end; ++it)
	{
		sum += *it != 0;
	}

	return (double)sum / (double)size;
}

