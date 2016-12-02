#include <user\UserManager.h>
#include <user\User.h>
#include <opencv2/imgproc.hpp>


#include <io/RequestHandler.h>
#include <io/RequestTypes.h>
#include <io/ResponseTypes.h>

#include <opencv2/highgui/highgui.hpp>

using namespace  user;

bool UserManager::Init(io::TCPClient* connection, io::NetworkRequestHandler* handler)
{
	if (connection == nullptr || handler == nullptr)
	{
		return false;
	}
	pServerConn = connection;
	pRequestHandler = handler;
	return true;
}


// refresh tracked users: scene_id, bounding boxes
void UserManager::RefreshTrackedUsers(const std::vector<int> &user_scene_ids, std::vector<cv::Rect2f> bounding_boxes)
{

	// add new user for all scene ids that are new
	for (int i = 0; i<user_scene_ids.size(); i++)
	{
		int scene_id = user_scene_ids[i];
		// user not tracked yet - initiate new user model
		if (mFrameIDToUser.count(scene_id) == 0)
		{
			mFrameIDToUser[scene_id] = new User();
		}
	}

	// update users infos - remove non tracked
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end();)
	{
		int user_index = find(user_scene_ids.begin(), user_scene_ids.end(), it->first) - user_scene_ids.begin();

		// check if user is in scene
		if (user_index < user_scene_ids.size())
		{
			// user is in scene - update scene data (bounding box, position etc.)
			it->second->SetFaceBoundingBox(bounding_boxes[user_index]);
			++it;
		}
		// remove user if he has left scene
		else
		{
			// remove request mapping
			RemovePointerMapping(it->second);

			// TODO: cancel requests

			// user has left scene - delete tracking instance
			delete(it->second);

#ifdef _DEBUG_USERMANAGER
			std::cout << "=== User has left scene - removing UserSceneID " << it->first << std::endl;
#endif

			// remove mapping
			mFrameIDToUser.erase(it++);	// increment after deletion
		}
	}
}

// incorporate processed requests: update user ids
void UserManager::ApplyUserIdentification()
{
	// handle all processed identification requests
	io::IdentificationResponse response;
	io::NetworkRequest* request = nullptr;

	while (pRequestHandler->PopResponse(&response, request))
	{
#ifdef _DEBUG_USERMANAGER
		std::cout << "--- Processing io::IdentificationResponse" << std::endl;
		// display response
		std::cout << "--- User ID: " << response.mUserID << std::endl;
#endif

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			// remove request mapping
			RemovePointerMapping(it->second);

			// check if other user in scene has same id
			// TODO: Handle Missdetection

			// apply user identification
			target_user->SetUserID(response.mUserID, response.mUserNiceName);
		}
		else {
			// user corresponding to request not found (may have left scene) - drop response

		}
	}

	// handle erronomous requests
	io::ErrorResponse err_response;
	io::NetworkRequestType req_type;
	while (pRequestHandler->PopResponse(&err_response, request, &req_type))
	{
		// display response
		std::cout << "--- Error response: " << err_response.mMessage << std::endl;

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request);

#ifdef _DEBUG_USERMANAGER
		std::cout << "--- RequestID (type): " << req_type << std::endl;
#endif

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;

			if (req_type == io::NetworkRequest_ImageIdentification) {
				target_user->SetIDStatus(user::IDStatus_Unknown);
			}

			// remove request mapping
			RemovePointerMapping(it->second);
			// reset user identification status if it was an identification request

		}
		else {
			// user corresponding to request not found (may have left scene) - drop response

		}
	}

}

// send identification requests for all unknown users
void UserManager::RequestUserIdentification(cv::Mat scene_rgb)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		if (it->second->GetIDStatus() == IDStatus_Unknown)
		{
			std::vector<cv::Mat> faces;
			// extract face patch
			cv::Mat face = scene_rgb(it->second->GetFaceBoundingBox());
			
			faces.push_back(face);

			//cv::imshow("face", face);
			//cv::waitKey(3);

			// make new identification request
			IDReq* new_request = new IDReq(pServerConn, faces);
			pRequestHandler->addRequest(new_request);

			// update linking
			mRequestToUser[new_request] = it->second;
			mUserToRequest[it->second] = new_request;

			// set user status
			it->second->SetIDStatus(IDStatus_Pending);
		}
	}
}

// ----------------- helper functions

void UserManager::DrawUsers(cv::Mat &img)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		cv::Rect bb = it->second->GetFaceBoundingBox();

		// draw face bounding box
		cv::rectangle(img, bb, cv::Scalar(0, 0, 255), 2, cv::LINE_4);

		// draw identification status
		float font_size = 0.5;
		std::string text;
		enum IdentificationStatus status = it->second->GetIDStatus();
		if (status == IDStatus_Identified)
		{
			int user_id = 0;
			std::string nice_name = "";
			it->second->GetUserID(user_id, nice_name);
			text = "Status: " + nice_name + " - ID" + std::to_string(user_id);
		}
		else if (status == IDStatus_Pending)
		{
			text = "Status: pending";
		}
		else
		{
			text = "Status: unknown";
		}
		cv::putText(img, text, cv::Point(bb.x+10, bb.y+20), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0, 0, 255), 1, 8);
	}
}
