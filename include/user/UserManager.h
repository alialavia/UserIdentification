#ifndef USER_USERMANAGER_H_
#define USER_USERMANAGER_H_

#include <map>
#include <set>
#include <vector>

#include <tracking\FaceTracker.h>
#include <opencv2/core.hpp>

// face aligner
#include <features\Face.h>

#define _DEBUG_USERMANAGER
//#define _DLIB_PREALIGN // use dlib on client side for face alignment

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

		UserManager() : 
		pServerConn(nullptr), 
		pRequestHandler(nullptr)
#ifdef _DLIB_PREALIGN
		,mDlibAligner(nullptr)
#endif
		{
		}
		~UserManager() {
#ifdef _DLIB_PREALIGN
			delete(mDlibAligner);
#endif

			// TODO: do proper cleanup
		}

		bool Init(io::TCPClient* connection, io::NetworkRequestHandler* handler);

		void RefreshUserTracking(
			const std::vector<int> &user_scene_ids, 
			std::vector<cv::Rect2f> bounding_boxes
		);

		// ------------ feature updaters

		// TODO: add templated UpdateFeatures function similar to RequestHandler
		void UpdateFaceData(std::vector<tracking::Face> faces, std::vector<int> user_ids);

		// ------------
		void ProcessResponses();
		void GenerateRequests(cv::Mat scene_rgb);

		// ----------------- helper functions
		void DrawUsers(cv::Mat &img);

		void RemoveRequestUserLinking(io::NetworkRequest* req) {
			// remove req->user mapping
			std::map<io::NetworkRequest*, User*>::iterator it1 = mRequestToUser.find(req);
			if (it1 != mRequestToUser.end()) {
				// User -> Request
				int succ = mUserToRequests.erase(it1->second);

#ifdef _DEBUG_USERMANAGER
				if (succ == 0) {
					throw std::invalid_argument("Request was not correctly linked with a user.");
				}
#endif
				// request -> User
				mRequestToUser.erase(it1);
			}
		}

		//void RemoveRequestUserLinking(User* user) {
		//	// delete all request linking for a user (request needs to be deleted manually if not poped from req/resp queue)

		//	
		//	std::map<User*, std::set<io::NetworkRequest*>>::iterator it1 = mUserToRequests.find(user);
		//	if (it1 != mUserToRequests.end()) {

		//		mUserToRequests.erase(it1);
		//	}


		//	std::map<User*, io::NetworkRequest*>::iterator it1 = mUserToRequest.find(user);
		//	if (it1 != mUserToRequest.end()) {
		//		std::map<io::NetworkRequest*, User*>::iterator it2 = mRequestToUser.find(it1->second);
		//		if (it2 != mRequestToUser.end()) {
		//			mRequestToUser.erase(it2);
		//		}
		//		mUserToRequest.erase(it1);
		//	}
		//}

		void CancelAllUserRequests(User* user);

	private:
		io::TCPClient* pServerConn;
		io::NetworkRequestHandler* pRequestHandler;

		// user identification requests
		// TODO: use smart pointers
		std::map<io::NetworkRequest*, User*> mRequestToUser;
		//std::map<User*, io::NetworkRequest*> mUserToRequest;
		std::map<User*, std::set<io::NetworkRequest*>> mUserToRequests;

		// scene id to user id mapping
		std::map<int, User*> mFrameIDToUser;

#ifdef _DLIB_PREALIGN
		// face aligner
		features::DlibFaceAligner* mDlibAligner;
#endif

	};

}

#endif