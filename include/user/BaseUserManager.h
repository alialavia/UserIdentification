#ifndef USER_BASEUSERMANAGER_H_
#define USER_BASEUSERMANAGER_H_

#include <map>
#include <set>
#include <vector>

#include <tracking\FaceTracker.h>
#include <opencv2/core.hpp>

// face aligner
#include <features\Face.h>

#define _DEBUG_USERMANAGER

namespace io {
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

	class BaseUserManager {
	public:

		BaseUserManager() :
			pServerConn(nullptr),
			pRequestHandler(nullptr)
#ifdef _DLIB_PREALIGN
			, mpDlibAligner(nullptr)
#endif
		{
		}
		~BaseUserManager() {
#ifdef _DLIB_PREALIGN
			delete(mpDlibAligner);
#endif
			// TODO: do proper cleanup
		}

		bool Init(io::TCPClient* connection, io::NetworkRequestHandler* handler);

		// To implement in specific managers (batch or stream)
		virtual void ProcessResponses() = 0;
		virtual void GenerateRequests(cv::Mat scene_rgb) = 0;

		/////////////////////////////////////////////////
		/// 	Core Methods

		void RefreshUserTracking(const std::vector<int> &user_scene_ids, std::vector<cv::Rect2f> bounding_boxes, std::vector<cv::Point3f> positions);
		void UpdateTrackingStatus();

		/////////////////////////////////////////////////
		/// 	Feature Updates

		// TODO: add templated UpdateFeatures function similar to RequestHandler
		void UpdateFaceData(std::vector<tracking::Face> faces, std::vector<int> user_ids);

		/////////////////////////////////////////////////
		/// 	Helpers

		void RenderGUI(cv::Mat &img);

		void RemoveRequestUserLinking(io::NetworkRequest* req) {
			// remove req->user mapping
			std::map<io::NetworkRequest*, User*>::iterator it1 = mRequestToUser.find(req);
			if (it1 != mRequestToUser.end()) {
				// User -> Request
				int succ = mUserToRequests[it1->second].erase(it1->first);
				//int succ = mUserToRequests.erase(it1->second);

#ifdef _DEBUG_USERMANAGER
				if (succ == 0) {
					throw std::invalid_argument("Request was not correctly linked with a user (1).");
				}
#endif
				// request -> User
				mRequestToUser.erase(it1);
			}
#ifdef _DEBUG_USERMANAGER
			else
			{
				throw std::invalid_argument("Request was not correctly linked with a user (2).");
			}
#endif
		}

		void CancelAndDropAllUserRequests(User* user);


		/////////////////////////////////////////////////
		/// 	API

		std::vector<std::pair<int, int>> GetUserandTrackingID();
		std::vector<std::pair<int, cv::Mat>> GetSceneProfilePictures();
		std::vector<std::pair<int, cv::Mat>> GetAllProfilePictures();
		void GetAllProfilePictures(std::vector<cv::Mat> &pictures, std::vector<int> &user_ids);
		bool GetUserID(const cv::Mat &face_capture, int &user_id);

	protected:
		io::TCPClient* pServerConn;
		io::NetworkRequestHandler* pRequestHandler;

		// user identification requests
		// TODO: use smart pointers
		std::map<io::NetworkRequest*, User*> mRequestToUser;
		//std::map<User*, io::NetworkRequest*> mUserToRequest;
		std::map<User*, std::set<io::NetworkRequest*>> mUserToRequests;

		// scene id to user mapping
		std::map<int, User*> mFrameIDToUser;

#ifdef _DLIB_PREALIGN
		// face aligner
		features::DlibFaceAligner* mpDlibAligner;
#endif

	};

}

#endif
