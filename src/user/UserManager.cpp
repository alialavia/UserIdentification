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

#ifdef _DLIB_PREALIGN
	// initialize aligner
	mDlibAligner = new features::DlibFaceAligner();
	mDlibAligner->Init();
#endif

	std::cout << "--- UserManager initialized" << std::endl;

	return true;
}

void UserManager::UpdateFaceData(std::vector<tracking::Face> faces, std::vector<int> user_ids) {
	for (size_t i = 0; i < faces.size(); i++) {
		User* u = mFrameIDToUser[user_ids[i]];
		u->SetFaceData(faces[i]);
	}
}

// refresh tracked users: scene_id, bounding boxes
void UserManager::RefreshUserTracking(
	const std::vector<int> &user_scene_ids, 
	std::vector<cv::Rect2f> bounding_boxes
)
{
	// add new user for all scene ids that are new
	for (int i = 0; i<user_scene_ids.size(); i++)
	{
		int scene_id = user_scene_ids[i];
		// user not tracked yet - initiate new user model
		if (mFrameIDToUser.count(scene_id) == 0)
		{
			// create new user
			mFrameIDToUser[scene_id] = new User();
		}
	}

	// update users infos - remove non tracked
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end();)
	{
		int user_frame_id = it->first;
		User* target_user = it->second;
		int user_index = find(user_scene_ids.begin(), user_scene_ids.end(), user_frame_id) - user_scene_ids.begin();

		// check if user is in scene
		if (user_index < user_scene_ids.size())
		{
			// user is in scene - update scene data (bounding box, position etc.)
			target_user->SetFaceBoundingBox(bounding_boxes[user_index]);
			// reset feature tracking
			target_user->ResetSceneFeatures();
			++it;
		}
		// remove user if he has left scene
		else
		{
			// cancel and unlink all pending requests for a user
			CancelAllUserRequests(target_user);

			// user has left scene - delete tracking instance
			delete(target_user);

#ifdef _DEBUG_USERMANAGER
			std::cout << "=== User has left scene - removing UserSceneID " << it->first << std::endl;
#endif

			// remove mapping
			mFrameIDToUser.erase(it++);	// increment after deletion
		}
	}
}

// incorporate processed requests: update user ids
void UserManager::ProcessResponses()
{

	io::NetworkRequest* request = nullptr;
	io::NetworkRequestType req_type;

	// ============================================= //
	// 1. handle identification responses
	// ============================================= //
	io::IdentificationResponse response;
	while (pRequestHandler->PopResponse(&response, request))
	{
#ifdef _DEBUG_USERMANAGER
		std::cout << "--- Processing io::IdentificationResponse" << std::endl;
		// display response
		std::cout << "--- User ID: " << response.mUserID << std::endl;
#endif

		if (request == nullptr) {
			std::cout << "--------------- USER NOT IN SCENE - REQUEST HAS BEEN DELETED ------------- " << std::endl;
			continue;
		}

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			io::NetworkRequest* target_request = it->first;

			// remove request mapping
			RemoveRequestUserLinking(target_request);

			bool duplicate_user = false;
			int duplicate_id = -1;

			// Check for duplicate IDs
			for (auto its = mFrameIDToUser.begin(); its != mFrameIDToUser.end(); ++its)
			{
				// check if has assigned id - equals identified
				if((duplicate_id = its->second->GetUserID()) > 0 && duplicate_id == response.mUserID)
				{
					// person with same id is already in scene
					// reset both tracking instances and force reidentification
					its->second->ResetUser();
					target_user->ResetUser();
					duplicate_user = true;
					break;
				}
			}
			
			if(duplicate_user)
			{
				std::cout << "-------------- DUPLICATE USER - RESETTING ID " << response.mUserID << std::endl;
				
			}else
			{
				// apply user identification
				target_user->SetUserID(response.mUserID, response.mUserNiceName);
			}

			// reset action status
			target_user->SetActionStatus(ActionStatus_Idle);
		}
		else {
			// user corresponding to request not found - nothing to unlink - drop response
			// e.g. User has left scene and all requests and linking where deleted
			throw std::invalid_argument("User unspecific requests are not implemented yet!");
		}
	}

	// ============================================= //
	// 2. Default successful tasks
	// ============================================= //
	io::ReidentificationResponse reid_response;
	while (pRequestHandler->PopResponse(&reid_response, request, &req_type))
	{
		// display response
		std::cout << "--- Forced reidentification (e.g. update does not explain model)" << std::endl;

		if (request == nullptr) {
			std::cout << "--------------- USER NOT IN SCENE - REQUEST HAS BEEN DELETED ------------- " << std::endl;
			continue;
		}

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			io::NetworkRequest* target_request = it->first;

			// remove request mapping
			RemoveRequestUserLinking(target_request);

			// reset target user
			target_user->ResetUser();
		}
		else {
			// user corresponding to request not found - nothing to unlink - drop response
			// e.g. User has left scene and all requests and linking where deleted
			throw std::invalid_argument("User unspecific requests are not implemented yet!");
		}
	}

	// ============================================= //
	// 3. Default successful tasks
	// ============================================= //
	io::OKResponse ok_response;
	while (pRequestHandler->PopResponse(&ok_response, request, &req_type))
	{
		// display response
		std::cout << "--- Ok response: " << ok_response.mMessage << std::endl;

		if (request == nullptr) {
			std::cout << "--------------- USER NOT IN SCENE - REQUEST HAS BEEN DELETED ------------- " << std::endl;
			continue;
		}

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			io::NetworkRequest* target_request = it->first;

			// remove request mapping
			RemoveRequestUserLinking(target_request);

			// reset action status
			target_user->SetActionStatus(ActionStatus_Idle);

			// handle custom events
			//if (req_type == io::NetworkRequest_EmbeddingCollectionByID) {
			//
			//}
		}
		else {
			// user corresponding to request not found - nothing to unlink - drop response
			// e.g. User has left scene and all requests and linking where deleted
			throw std::invalid_argument("User unspecific requests are not implemented yet!");
		}
	}

	// ============================================= //
	// 4. Default erronomous tasks
	// ============================================= //
	io::ErrorResponse err_response;
	while (pRequestHandler->PopResponse(&err_response, request, &req_type))
	{
		// display response
		std::cout << "--- Error response | RequestID (" << req_type  << "): " << err_response.mMessage << std::endl;

		if (request == nullptr) {
			std::cout << "--------------- USER NOT IN SCENE - REQUEST HAS BEEN DELETED ------------- " << std::endl;
			continue;
		}

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			io::NetworkRequest* target_request = it->first;

			// error during identification (e.g. no faces detected) - try again
			if (req_type == io::NetworkRequest_ImageIdentification) {
				target_user->SetIDStatus(user::IDStatus_Unknown);
				target_user->SetActionStatus(ActionStatus_Idle);
			}
			// error during update - not enough "good"/destinctive feature vectors (most vectors are around threshold)
			// trash update and start again
			else if (req_type == io::NetworkRequest_EmbeddingCollectionByIDAligned
				|| io::NetworkRequest_EmbeddingCollectionByID
				|| io::NetworkRequest_EmbeddingCollectionByName) 
			{
				target_user->SetActionStatus(ActionStatus_Idle);
			}

			// remove request mapping
			RemoveRequestUserLinking(target_request);
			// reset user identification status if it was an identification request

		}
		else {
			// user corresponding to request not found (may have left scene or the request is not user specific) - drop response
			// unprocessed requests are already deleted when user leaves scene
			throw std::invalid_argument("User unspecific requests are not implemented yet!");
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

		//std::cout << "--- id_Status: "<< id_status << " | action: "<< action << std::endl;

		// request user identification
		if (id_status == IDStatus_Unknown)
		{

			// new user in scene
			if (action == ActionStatus_Idle) {
				it->second->SetActionStatus(ActionStatus_Initialization);
				action = ActionStatus_Initialization;
			}

			// collect images for identification
			if (action == ActionStatus_Initialization) {



#ifdef FACEGRID_RECORDING
				// check if face should be recorded
				tracking::Face face;
				if (it->second->GetFaceData(face)) {

					// collect another image
					cv::Mat face_snap = scene_rgb(it->second->GetFaceBoundingBox());

					// resize
					cv::resize(face_snap, face_snap, cv::Size(96, 96));

					int roll, pitch, yaw;
					face.GetEulerAngles(roll, pitch, yaw);
					try
					{
						// add face if not yet capture from this angle
						if (it->second->pGrid->IsFree(roll, pitch, yaw)) {
							it->second->pGrid->StoreSnapshot(roll, pitch, yaw, face_snap);
							std::cout << "-- take snapshot" << std::endl;
						}
					}
					catch (...)
					{
					}

					// if enough images, request identification
					if (it->second->pGrid->nr_images() > 9) {

						// extract images
						std::vector<cv::Mat*> face_patches = it->second->pGrid->ExtractGrid();

						// make new identification request
						IDReq* new_request = new IDReq(pServerConn, face_patches);
						pRequestHandler->addRequest(new_request);

						// update linking
						mRequestToUser[new_request] = it->second;
						mUserToRequests[it->second].insert(new_request);

						// set user action status
						it->second->SetActionStatus(ActionStatus_IDPending);
						it->second->pGrid->Clear();
					}
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
			// send model updates - reinforced learning
			// TODO: implement
			if (action == ActionStatus_Idle) {
				it->second->SetActionStatus(ActionStatus_DataCollection);
				action = ActionStatus_DataCollection;
			}

			if (action == ActionStatus_DataCollection) {


#ifdef FACEGRID_RECORDING


				// check if face should be recorded
				tracking::Face face;
				if (it->second->GetFaceData(face)) {

					int roll, pitch, yaw;
					face.GetEulerAngles(roll, pitch, yaw);

					// face from this pose not yet recorded
					if (it->second->pGrid->IsFree(roll, pitch, yaw)) {

						cv::Rect2f facebb = it->second->GetFaceBoundingBox();
						// TODO: debug why face bb is nan
						std::cout << "............ Face bb: " << facebb.height << " | " << facebb.width << std::endl;

						// collect another image
						cv::Mat face_snap = scene_rgb(facebb);

						// detect and warp face
						cv::Mat aligned;
						
#ifdef _DLIB_PREALIGN
							if (mDlibAligner->AlignImage(96, face_snap, aligned)) 

#else
						aligned = face_snap;
						// resize (requests needs all squared with same size)
						cv::resize(aligned, aligned, cv::Size(120, 120));
#endif
							{
							std::cout << "cols: " << aligned.cols << std::endl;

								// save
								try
								{
									// add face if not yet capture from this angle
									it->second->pGrid->StoreSnapshot(roll, pitch, yaw, aligned);
									std::cout << "--- take snapshot: " << it->second->pGrid->nr_images() << std::endl;
									/*cv::imshow("aligned", aligned);
									cv::waitKey(2);*/

							/*		cv::Mat grid;
									it->second->pGrid->GetFaceGridPitchYaw(grid);
									cv::imshow("Grid", grid);
									cv::waitKey(3);*/
								}
								catch (...)
								{
								}
							}



					}	// /free pose position

						// if enough images, request identification
					if (it->second->pGrid->nr_images() > 9) {

						// extract images
						std::vector<cv::Mat*> face_patches = it->second->pGrid->ExtractGrid();

						if(face_patches[0]->cols == 0)
						{
							std::cout << "----------- WHY? -----------" << std::endl;
						}
						int user_id; std::string user_name;

						// TODO: DEBUG HERE
						// ID -1
						it->second->GetUserID(user_id, user_name);

						// make new identification request
#ifdef _DLIB_PREALIGN
						io::EmbeddingCollectionByIDAligned* new_request = new io::EmbeddingCollectionByIDAligned(pServerConn, face_patches, user_id);
#else
						io::EmbeddingCollectionByID* new_request = new io::EmbeddingCollectionByID(pServerConn, face_patches, user_id);
#endif
						pRequestHandler->addRequest(new_request);

						// update linking
						mRequestToUser[new_request] = it->second;
						mUserToRequests[it->second].insert(new_request);

						// set user action status
						it->second->SetActionStatus(ActionStatus_UpdatePending);
						it->second->pGrid->Clear();
					}
				}	//	/end face data available
#endif

			}
			else if (action == ActionStatus_UpdatePending) {
				// do nothing
			}


		}
		
	}

}

void UserManager::CancelAllUserRequests(User* user) {
	// user->requests
	if (mUserToRequests.find(user) != mUserToRequests.end()) {

		// iterate over requests
		for (auto req : mUserToRequests[user]) {
			// unlink: requests->user
			mRequestToUser.erase(req);
			// delete: requests from queue
			pRequestHandler->cancelRequest(req);
		}
		// unlink: user->request
		mUserToRequests.erase(user);
	}
	else {
		// not found
	}
}

// ----------------- helper functions

void UserManager::DrawUsers(cv::Mat &img)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		cv::Rect bb = it->second->GetFaceBoundingBox();

		// draw identification status
		float font_size = 0.5;
		cv::Scalar color = cv::Scalar(0, 0, 0);
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
			color = cv::Scalar(0, 255, 0);
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
			color = cv::Scalar(0, 0, 255);
			
		}

		int baseline = 0;
		cv::Size textSize = cv::getTextSize(text1, cv::FONT_HERSHEY_SIMPLEX, font_size, 1, &baseline);

		cv::Rect bg_patch = cv::Rect(bb.x, bb.y, textSize.width + 20, textSize.height + 15);
		img(bg_patch) = cv::Scalar(0, 0, 0);

		cv::putText(img, text1, cv::Point(bb.x+10, bb.y+20), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1, 8);
		cv::putText(img, text2, cv::Point(bb.x+10, bb.y+40), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1, 8);

		// draw face bounding box
		cv::rectangle(img, bb, color, 2, cv::LINE_4);
	}
}
