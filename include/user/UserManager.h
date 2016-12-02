#ifndef USER_USERMANAGER_H_
#define USER_USERMANAGER_H_

#include <map>
#include <vector>
#include <opencv2/core.hpp>

#define _DEBUG_USERMANAGER

namespace io{
class TCPClient;
class NetworkRequestHandler;
class ImageIdentification;
class NetworkRequest;
}

namespace user
{

	typedef io::ImageIdentification IDReq;

	class User;

	/*
	In each loop:
	- refresh tracked users
	- update user ids (from processed requests)
	- request identification for unknown users
	*/

	class UserManager {
	public:

		UserManager() : pServerConn(nullptr), pRequestHandler(nullptr)
		{
		}

		bool Init(io::TCPClient* connection, io::NetworkRequestHandler* handler);

		void RefreshTrackedUsers(const std::vector<int> &user_scene_ids, std::vector<cv::Rect2f> bounding_boxes);
		void ApplyUserIdentification();
		void RequestUserIdentification(cv::Mat scene_rgb);

		// ----------------- helper functions
		void DrawUsers(cv::Mat &img);

		void RemovePointerMapping(io::NetworkRequest* id_req) {
			std::map<io::NetworkRequest*, User*>::iterator it1 = mRequestToUser.find(id_req);
			if (it1 != mRequestToUser.end()) {
				std::map<User*, io::NetworkRequest*>::iterator it2 = mUserToRequest.find(it1->second);
				if (it2 != mUserToRequest.end()) {
					mUserToRequest.erase(it2);
				}
				mRequestToUser.erase(it1);
			}
		}

		void RemovePointerMapping(User* user) {

			std::map<User*, io::NetworkRequest*>::iterator it1 = mUserToRequest.find(user);
			if (it1 != mUserToRequest.end()) {
				std::map<io::NetworkRequest*, User*>::iterator it2 = mRequestToUser.find(it1->second);
				if (it2 != mRequestToUser.end()) {
					mRequestToUser.erase(it2);
				}
				mUserToRequest.erase(it1);
			}
		}

	private:
		io::TCPClient* pServerConn;
		io::NetworkRequestHandler* pRequestHandler;

		// user identification requests
		// TODO: use smart pointers
		std::map<io::NetworkRequest*, User*> mRequestToUser;
		std::map<User*, io::NetworkRequest*> mUserToRequest;

		// frame id to user id mapping
		std::map<int, User*> mFrameIDToUser;

	};

}

#endif