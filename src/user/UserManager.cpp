#include <user\UserManager.h>
#include <user\User.h>
#include <opencv2/imgproc.hpp>


#include <io/RequestHandler.h>
#include <io/RequestTypes.h>
#include <io/ResponseTypes.h>

#include <opencv2/highgui/highgui.hpp>

#include <tracking\FaceTracker.h>

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
void UserManager::RefreshTrackedUsers(
	const std::vector<int> &user_scene_ids, 
	std::vector<cv::Rect2f> bounding_boxes, 
	std::vector<tracking::Face> faces
)
{

	// add new user for all scene ids that are new
	for (int i = 0; i<user_scene_ids.size(); i++)
	{
		int scene_id = user_scene_ids[i];
		// user not tracked yet - initiate new user model
		if (mFrameIDToUser.count(scene_id) == 0)
		{
			User* u = new User();


			// TODO: DEBUG

			IdentificationStatus id_status = IDStatus_Unknown;
			ActionStatus action = ActionStatus_Idle;
			u->GetStatus(id_status, action);
			//std::cout << "----- user init: " << static_cast<int>(id_status) << " | action: " << action << std::endl;
			throw 20;


			mFrameIDToUser[scene_id] = u;


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
			// update face data
			it->second->SetFaceData(faces[user_index]);
			++it;
#ifdef _DEBUG_USERMANAGER
			std::cout << "=== update user " << std::endl;
#endif
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
			target_user->SetActionStatus(ActionStatus_Idle);
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
				target_user->SetActionStatus(ActionStatus_Idle);
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
void UserManager::GenerateRequests(cv::Mat scene_rgb)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		IdentificationStatus id_status;
		ActionStatus action;
		it->second->GetStatus(id_status, action);

		std::cout << "--- id_Status: "<< id_status << " | action: "<< action << std::endl;

		// request user identification
		if (id_status == IDStatus_Unknown)
		{

			std::cout << "--- ¨1INITIALZATION. ID status: " << id_status << " - action: "<< action << std::endl;

			// new user in scene
			if (action == ActionStatus_Idle) {
				it->second->SetActionStatus(ActionStatus_Initialization);
				action = ActionStatus_Initialization;
				std::cout << "--- .INITIALZATION" << std::endl;
			}

			// collect images for identification
			if (action == ActionStatus_Initialization) {

				std::cout << "--- INITIALZATION" << std::endl;
				// collect another image
				cv::Mat face_snap = scene_rgb(it->second->GetFaceBoundingBox());

#ifdef FACEGRID_RECORDING
				// check if face should be recorded
				tracking::Face face = it->second->GetFaceData();
				int roll, pitch, yaw;
				face.GetEulerAngles(roll, pitch, yaw);
				try
				{
					// add face if not yet capture from this angle
					if (it->second->pGrid->IsFree(roll, pitch, yaw)) {
						it->second->pGrid->StoreSnapshot(roll, pitch, yaw, face_snap);
						
					}
				}
				catch (...)
				{
				}

				// if enough images, request identification
				if (it->second->pGrid->nr_images() > 5) {

					// extract images
					std::vector<cv::Mat*> face_patches = it->second->pGrid->ExtractGrid();

					// make new identification request
					IDReq* new_request = new IDReq(pServerConn, face_patches);
					pRequestHandler->addRequest(new_request);

					// update linking
					mRequestToUser[new_request] = it->second;
					mUserToRequest[it->second] = new_request;

					// set user action status
					it->second->SetActionStatus(ActionStatus_IDPending);
				}
#endif

			}			
			// wait for identification response
			else if (action == ActionStatus_IDPending) {
				// do nothing
			}


			//cv::imshow("face", face);
			//cv::waitKey(3);

		}
		else if (id_status == IDStatus_Identified) {
			// send model updates



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
		std::string text1, text2;

		IdentificationStatus id_status;
		ActionStatus action;
		it->second->GetStatus(id_status, action);

		if (id_status == IDStatus_Identified)
		{
			int user_id = 0;
			std::string nice_name = "";
			it->second->GetUserID(user_id, nice_name);
			text1 = "Status: " + nice_name + " - ID" + std::to_string(user_id);
		}
		else
		{
			text1 = "Status: unknown";
			if (action == ActionStatus_Initialization) {
				text2 = "Initialization";
			}
			else if (action == ActionStatus_IDPending) {
				text2 = "ID pending";
			}
			else if (action == ActionStatus_Idle) {
				text2 = "Idle";
			}
			
		}
		cv::putText(img, text1, cv::Point(bb.x+10, bb.y+20), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0, 0, 255), 1, 8);
		cv::putText(img, text2, cv::Point(bb.x+10, bb.y+40), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0, 0, 255), 1, 8);
	}
}
