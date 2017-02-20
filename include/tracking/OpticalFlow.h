#ifndef TRACKING_OPTICALFLOW_H_
#define TRACKING_OPTICALFLOW_H_

#include <vector>
#include <opencv2\opencv.hpp>
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
//#include "opencv2\objdetect\objdetect.hpp"
//#include "opencv2/video/tracking.hpp"



namespace tracking
{

	class OpticalFlow
	{
	public:
		OpticalFlow(int resize_width=200): mFlowThresh(0.7), mWidth(resize_width){

		}
		~OpticalFlow() {

		}
		void UpdateFlow(const cv::Mat &img_color) {

			cv::Mat tmp = img_color.clone();

			// scale
			ResizeMinSize(tmp);

			// convert to grayscale
			if (tmp.channels() == 3) {
				cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
			}

			// blur (remove high frequency noise)
			cv::GaussianBlur(tmp, tmp, cv::Size(5,5), 0, 0);

			if (mPrevGray.empty()) {
				tmp.copyTo(mPrevGray);
				return;
			}

			//std::cout << "Prev: Channels: " << mPrevGray.channels() << std::endl;
			//std::cout << "Prev: Size: " << mPrevGray.size() << std::endl;
			//std::cout << "curr: Channels: " << tmp.channels() << std::endl;
			//std::cout << "curr: Size: " << tmp.size() << std::endl;

			// calculate optical flow 
			calcOpticalFlowFarneback(mPrevGray, tmp, mFlowMat, 0.4, 1, 20, 2, 8, 1.2, 0);
		

			// fill previous image again
			tmp.copyTo(mPrevGray);
		}

		double GetAbsoluteFlow() {

			// calculate summed flow
			if (mFlowMat.empty()) {
				return 0.;
			}
			//double sum = cv::sum(mFlowMat)[0];

			cv::Mat flow;
			mFlowMat.copyTo(flow);

			double sum = 0.;
			for (int y = 0; y < mPrevGray.rows; y += 5) {
				for (int x = 0; x < mPrevGray.cols; x += 5)
				{
					// get the flow from y, x position * 10 for better visibility
					cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x);
					double n = cv::norm(cv::Mat(flowatxy));
					if (n>mFlowThresh) {
						sum += n;
					}
				}
			}

			return sum;
		}

		void DispFlow() {

			if (mFlowMat.empty()) {
				return;
			}

			cv::Mat original, flow;
			mPrevGray.copyTo(original);
			mFlowMat.copyTo(flow);

			// By y += 5, x += 5 you can specify the grid 
			for (int y = 0; y < mPrevGray.rows; y += 5) {
				for (int x = 0; x < original.cols; x += 5)
				{
					// get the flow from y, x position * 10 for better visibility
					const cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x) * 10;
					// draw line at flow direction
					line(original, cv::Point(x, y), cv::Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), cv::Scalar(255, 0, 0));
					// draw initial point
					circle(original, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
				}
			
			}
			
			// draw the results
			imshow("prew", original);
		}
		

	private:

		void ResizeMinSize(cv::Mat &to_resize) {

			// scale previously calculated
			if (mRescaleWidth != 0) {
				cv::resize(to_resize, to_resize, cv::Size(mRescaleWidth, mRescaleHeight));
				return;
			}

			if (mWidth > 0) {
				mRescaleWidth = mWidth;
				mRescaleHeight = (float)mMaxHeight / (float)to_resize.rows * (float)to_resize.rows;
			}
			else {
				// dynamic resizing
				// no resizing necessary
				if (to_resize.cols <= mMaxWidth && to_resize.rows <= mMaxHeight) {
					mRescaleWidth = to_resize.cols;
					mRescaleHeight = to_resize.rows;
					return;
				}

				float scale = (float)mMaxWidth / (float)to_resize.cols;
				int resizeWidth = mMaxWidth;
				int resizeHeight = scale * to_resize.rows;
				if (scale * to_resize.rows > mMaxHeight) {
					// scale in height
					resizeHeight = mMaxHeight;
					scale = (float)mMaxHeight / (float)to_resize.rows;
					resizeWidth = scale * to_resize.cols;
				}
				else {
					// scale in width

				}
				mRescaleWidth = resizeWidth;
				mRescaleHeight = resizeHeight;
			}

			cv::resize(to_resize, to_resize, cv::Size(mRescaleWidth, mRescaleHeight));

		}

		// settings
		float mFlowThresh;

		// dynamic size
		int mMaxWidth = 250;
		int mMaxHeight = 210;
		// static size
		int mWidth;

		// store scale
		int mRescaleWidth = 0;
		int mRescaleHeight = 0;

		// data containers
		cv::UMat mFlowMat;
		cv::UMat mPrevGray;

	};


} // namespace

#endif
