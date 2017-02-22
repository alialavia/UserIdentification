#ifndef USER_STREAMUSERMANAGER_H_
#define USER_STREAMUSERMANAGER_H_

#include <map>
#include <set>
#include <vector>

#include <user\BaseUserManager.h>
#include <opencv2/core.hpp>

#define _DEBUG_STREAMUSERMANAGER

namespace io {
	class ImageIdentification;
}

namespace user
{
	class User;

	/*
	In each loop:
	- refresh tracked users
	- update user ids (from processed requests)
	- request identification for unknown users
	*/

	class StreamUserManager : public BaseUserManager {
	public:

		StreamUserManager() : BaseUserManager() {
		}
		~StreamUserManager() {
		}

		/////////////////////////////////////////////////
		/// 	Core Methods

		void ProcessResponses();
		void GenerateRequests(cv::Mat scene_rgb);

	};

}

#endif
