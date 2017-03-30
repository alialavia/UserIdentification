#ifndef GUI_GUI_H_
#define GUI_GUI_H_

#include <opencv2\opencv.hpp>

namespace gui {

	void safe_copyTo(cv::Mat &target, const cv::Mat &input, cv::Rect dst) {

		cv::Mat img_copy = input;

		// resize
		if (input.cols != dst.width || input.rows != dst.height) {
			cv::resize(img_copy, img_copy, cv::Size(dst.width, dst.height));
		}

		if (dst.y < target.rows && dst.x < target.cols)
		{
			// check if roi overlapps image borders
			cv::Rect target_roi = dst;
			cv::Rect src_roi = cv::Rect(0, 0, img_copy.cols, img_copy.rows);

			if (target_roi.y < 0)
			{
				src_roi.y = -target_roi.y;
				src_roi.height += target_roi.y;
				target_roi.height += target_roi.y;
				target_roi.y = 0;
			}

			if (target_roi.x + img_copy.cols > target.cols)
			{
				src_roi.width = target_roi.x + img_copy.cols - target.cols;
				target_roi.width = target_roi.x + img_copy.cols - target.cols;
			}

			try {
				img_copy = img_copy(src_roi);
				img_copy.copyTo(target(target_roi));
			}
			catch (...) {
				// ...
			}

		}

	}


} // namespace

#endif