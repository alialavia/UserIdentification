#ifndef RECOGNITION_LOWERBODYRECOGNITION_H_
#define RECOGNITION_LOWERBODYRECOGNITION_H_


#include <opencv2\opencv.hpp>
#include <Windows.h>

#include <base\UserIdentification.h>

// Kinect SDK 2
//#include <Kinect.h>
//#include <vector>

namespace recognition
{
	class LowerBodyDescriptor {
	public:
		LowerBodyDescriptor() {

		}

		void ExtractFeatures(std::vector<std::vector<cv::Point2f>> lb_points, cv::Mat user_image) {
			// extract pixels


			// build color histogram

		}


	};

} // namespace

#endif
