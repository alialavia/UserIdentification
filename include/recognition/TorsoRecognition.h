#ifndef RECOGNITION_TORSORECOGNITION_H_
#define RECOGNITION_TORSORECOGNITION_H_


#include <opencv2\opencv.hpp>

// Kinect SDK 2
//#include <Kinect.h>
//#include <vector>

namespace recognition
{
	class TorsoDescriptor {
	public:
		TorsoDescriptor() {

		}

		void ExtractFeatures(std::vector<std::vector<cv::Point2f>> torso_points, cv::Mat user_image) {
			// search for sift features


		}

	};

} // namespace

#endif
