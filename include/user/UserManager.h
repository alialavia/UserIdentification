#ifndef USER_USERMANAGER_H_
#define USER_USERMANAGER_H_

#include <map>
#include <vector>
#include <opencv2/core.hpp>


namespace io{
class TCPClient;
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


	class UserManager {
	public:

		UserManager() : pServerConn(nullptr)
		{
		}

		bool Init(io::TCPClient* connection);

		void RefreshTrackedUsers(const std::vector<int> &user_scene_ids);
		void ApplyUserIdentification();
		void RequestUserIdentification();

		// ----------------- helper functions

		void DrawUsers(cv::Mat &img);

	private:
		io::TCPClient* pServerConn;
		std::map<int, User*> mFrameIDToUser;

	};

}

#endif