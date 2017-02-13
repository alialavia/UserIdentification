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
			// already processed requests are ignored when response is popped from request handler
			CancelAndDropAllUserRequests(target_user);

			// user has left scene - delete tracking instance
			delete(target_user);

#ifdef _DEBUG_USERMANAGER
			std::cout << "=== User has left scene - removing UserSceneID " << it->first << std::endl;
#endif

			// remove mapping
			mFrameIDToUser.erase(it++);	// increment after deletion
		}
	}

#ifdef _CHECK_BB_SWAP
	// update the tracking safety status
	UpdateTrackingSafetyMeasure();
#endif

}

// incorporate processed requests: update user ids
void UserManager::ProcessResponses()
{

	io::NetworkRequest* request_lookup = nullptr;	// careful! the request corresponding to this pointer is already deleted!
	io::NetworkRequestType req_type;

	// ============================================= //
	// 1. handle identification responses
	// ============================================= //
	io::IdentificationResponse response;
	while (pRequestHandler->PopResponse(&response, request_lookup))
	{
#ifdef _DEBUG_USERMANAGER
		std::cout << "--- Processing io::IdentificationResponse" << std::endl;
		// display response
		std::cout << "--- User ID: " << response.mUserID << std::endl;
#endif

		// check if response can be allocated with a request or if it should be dropped
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request_lookup);

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
					// cancel all requests for the two users
					CancelAndDropAllUserRequests(its->second);
					CancelAndDropAllUserRequests(target_user);

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
				// profile picture
				if(!response.mImage.empty())
				{
					target_user->AssignProfilePicture(response.mImage);
				}else
				{
					//std::cout << "------ no profile picture was taken before\n";
				}
				// update confidence
				target_user->SetConfidence(response.mConfidence);
			}

			// reset action status
			target_user->SetPendingProfilePicture(false);
			target_user->SetActionStatus(ActionStatus_Idle);
		}
		else {
			// user corresponding to request not found - nothing to unlink - drop response
			// e.g. User has left scene and all requests and linking where deleted
			//throw std::invalid_argument("User unspecific requests are not implemented yet!");

			std::cout << "------ USER HAS LEFT SCENE. IGNORE THIS RESPONSE." << std::endl;
		}
	}

	// ============================================= //
	// 2. Reidentification
	// ============================================= //
	io::ReidentificationResponse reid_response;
	while (pRequestHandler->PopResponse(&reid_response, request_lookup, &req_type))
	{
		// display response
		std::cout << "--- Forced reidentification (e.g. update does not explain model)" << std::endl;

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request_lookup);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			io::NetworkRequest* target_request = it->first;

			// remove request mapping
			RemoveRequestUserLinking(target_request);

			// calcel all requests
			CancelAndDropAllUserRequests(target_user);

			// reset target user
			target_user->ResetUser();
		}
		else {
			// user corresponding to request not found - nothing to unlink - drop response
			// e.g. User has left scene and all requests and linking where deleted
			//throw std::invalid_argument("User unspecific requests are not implemented yet!");
			std::cout << "------ USER HAS LEFT SCENE. IGNORE THIS RESPONSE." << std::endl;
		}
	}


	// ============================================= //
	// Update response
	// ============================================= //
	io::UpdateResponse update_r;
	while (pRequestHandler->PopResponse(&update_r, request_lookup, &req_type))
	{
		// display response
		std::cout << "--- Update response" << std::endl;

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request_lookup);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			io::NetworkRequest* target_request = it->first;

			// remove request mapping
			RemoveRequestUserLinking(target_request);

			// update confidence
			target_user->SetConfidence(update_r.mConfidence);
			// reset action status
			target_user->SetActionStatus(ActionStatus_Idle);
		}
		else {
			// user corresponding to request not found - nothing to unlink - drop response
			// e.g. User has left scene and all requests and linking where deleted
			std::cout << "------ USER HAS LEFT SCENE. IGNORE THIS RESPONSE." << std::endl;
		}
	}

	// ============================================= //
	// Profile Picture Update
	// ============================================= //
	io::QuadraticImageResponse img_response;
	while (pRequestHandler->PopResponse(&img_response, request_lookup, &req_type))
	{
		// display response
		std::cout << "--- Image response" << std::endl;

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request_lookup);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			io::NetworkRequest* target_request = it->first;

			// remove request mapping
			RemoveRequestUserLinking(target_request);

			// check request type
			if(req_type == io::NetworkRequest_ProfilePictureUpdate)
			{
				// update profile picture
				target_user->AssignProfilePicture(img_response.mImage);
				target_user->SetPendingProfilePicture(false);
			}
		}
		else {
			// user corresponding to request not found - nothing to unlink - drop response
			// e.g. User has left scene and all requests and linking where deleted
			std::cout << "------ USER HAS LEFT SCENE. IGNORE THIS RESPONSE." << std::endl;
		}
	}


	// ============================================= //
	// 3. Default successful tasks
	// ============================================= //
	io::OKResponse ok_response;
	while (pRequestHandler->PopResponse(&ok_response, request_lookup, &req_type))
	{
		// display response
		std::cout << "--- Ok response: " << ok_response.mMessage << std::endl;

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request_lookup);

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
			//throw std::invalid_argument("User unspecific requests are not implemented yet!");
			std::cout << "------ USER HAS LEFT SCENE. IGNORE THIS RESPONSE." << std::endl;
		}
	}

	// ============================================= //
	// 4. Default erronomous tasks
	// ============================================= //
	io::ErrorResponse err_response;
	while (pRequestHandler->PopResponse(&err_response, request_lookup, &req_type))
	{
		// display response
		std::cout << "--- Error response | RequestID (" << req_type  << "): " << err_response.mMessage << std::endl;

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request_lookup);

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

		}
		else {
			// user corresponding to request not found (may have left scene or the request is not user specific) - drop response
			// unprocessed requests are already deleted when user leaves scene
			//throw std::invalid_argument("User unspecific requests are not implemented yet!");
			std::cout << "------ USER HAS LEFT SCENE. IGNORE THIS RESPONSE." << std::endl;
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
		user::User* target_user = it->second;
		target_user->GetStatus(id_status, action);

		//std::cout << "--- id_Status: "<< id_status << " | action: "<< action << std::endl;

		// request user identification
		if (id_status == IDStatus_Unknown)
		{

			// new user in scene
			if (action == ActionStatus_Idle) {
				target_user->SetActionStatus(ActionStatus_Initialization);
				action = ActionStatus_Initialization;
			}

			// collect images for identification
			if (action == ActionStatus_Initialization) {

#ifdef FACEGRID_RECORDING
				// check if face should be recorded
				tracking::Face face;
				if (target_user->GetFaceData(face)) {

					int roll, pitch, yaw;
					face.GetEulerAngles(roll, pitch, yaw);
					// face from this pose not yet recorded
					if (target_user->pGrid->IsFree(roll, pitch, yaw)) {
						cv::Rect2f facebb = target_user->GetFaceBoundingBox();
						cv::Mat face_snap = scene_rgb(facebb);
						cv::Mat aligned;
#ifdef _DLIB_PREALIGN
						if (mDlibAligner->AlignImage(96, face_snap, aligned))
#else
						aligned = face_snap;
						// resize (requests needs all squared with same size)
						cv::resize(aligned, aligned, cv::Size(120, 120));
#endif
						{
							try
							{
								target_user->pGrid->StoreSnapshot(roll, pitch, yaw, aligned);
							}
							catch (...)
							{
							}
						}
					}	// /free pose position




					// if enough images, request identification
					if (target_user->pGrid->nr_images() > 9) {

						// extract images
						std::vector<cv::Mat*> face_patches = target_user->pGrid->ExtractGrid();

						// make new identification request
#ifdef _DLIB_PREALIGN
						io::ImageIdentificationAligned* new_request = new io::ImageIdentificationAligned(pServerConn, face_patches);
#else
						IDReq* new_request = new IDReq(pServerConn, face_patches);
#endif
						pRequestHandler->addRequest(new_request);

						// update linking
						mRequestToUser[new_request] = target_user;
						mUserToRequests[target_user].insert(new_request);

						// set user action status
						target_user->SetPendingProfilePicture(true);
						target_user->SetActionStatus(ActionStatus_IDPending);
						target_user->pGrid->Clear();
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
			
			// Update/assign profile picture
			if(target_user->NeedsProfilePicture() &&
				target_user->IsViewedFromFront())
			{
				target_user->SetPendingProfilePicture(true);
				cv::Mat profile_picture = scene_rgb(target_user->GetFaceBoundingBox());
				// scale
				cv::resize(profile_picture, profile_picture, cv::Size(120, 120));
				// make request
				io::ProfilePictureUpdate* new_request = new io::ProfilePictureUpdate(pServerConn, target_user->GetUserID(), profile_picture);
				pRequestHandler->addRequest(new_request);
				// update linking
				mRequestToUser[new_request] = target_user;
				mUserToRequests[target_user].insert(new_request);
			}


			if (action == ActionStatus_Idle) {
				target_user->SetActionStatus(ActionStatus_DataCollection);
				action = ActionStatus_DataCollection;
			}

			// send model updates - reinforced learning
			if (action == ActionStatus_DataCollection) {
#ifdef FACEGRID_RECORDING
				// check if face should be recorded
				tracking::Face face;
				if (target_user->GetFaceData(face)) {

					int roll, pitch, yaw;
					face.GetEulerAngles(roll, pitch, yaw);

					// face from this pose not yet recorded
					if (target_user->pGrid->IsFree(roll, pitch, yaw)) {

						cv::Rect2f facebb = target_user->GetFaceBoundingBox();

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
								// save
								try
								{
									// add face if not yet capture from this angle
									target_user->pGrid->StoreSnapshot(roll, pitch, yaw, aligned);
									//std::cout << "--- take snapshot: " << target_user->pGrid->nr_images() << std::endl;
								}
								catch (...)
								{
								}
							}
					}	// /free pose position

						// if enough images, request identification
					if (target_user->pGrid->nr_images() > 9) {

						// extract images
						std::vector<cv::Mat*> face_patches = target_user->pGrid->ExtractGrid();

						if (face_patches[0]->cols == 0)
						{
							std::cout << "----------- WHY? -----------" << std::endl;
						}
						int user_id; std::string user_name;

						// TODO: DEBUG HERE
						// ID -1
						target_user->GetUserID(user_id, user_name);

						io::EmbeddingCollectionByID* new_request;
#ifdef _DLIB_PREALIGN
#ifdef _CHECK_BB_SWAP

					// robust update: check update for model consistency
					if (!target_user->TrackingIsSafe()) {
						new_request = new io::EmbeddingCollectionByID(
							pServerConn, face_patches, user_id,
							io::NetworkRequest_EmbeddingCollectionByIDAlignedRobust	// specified request type
						);
					}else
#endif
					
						{new_request = new io::EmbeddingCollectionByID(
							pServerConn, face_patches, user_id, 
							io::NetworkRequest_EmbeddingCollectionByIDAligned	// specified request type
						);}

#else
						// standard update
						new_request = new io::EmbeddingCollectionByID(pServerConn, face_patches, user_id);
#endif
						pRequestHandler->addRequest(new_request);

						// update linking
						mRequestToUser[new_request] = target_user;
						mUserToRequests[target_user].insert(new_request);

						// set user action status
						target_user->SetActionStatus(ActionStatus_UpdatePending);
						target_user->pGrid->Clear();
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

void UserManager::CancelAndDropAllUserRequests(User* user) {
	// delete pending requests and delete all linking
	// processed requests without linking are dropped

	// user->requests
	if (mUserToRequests.find(user) != mUserToRequests.end()) {

		// iterate over requests
		for (auto req : mUserToRequests[user]) {
			// unlink: requests->user
			mRequestToUser.erase(req);
			// delete: requests from queue
			pRequestHandler->cancelPendingRequest(req);
		}
		// unlink: user->request
		mUserToRequests.erase(user);
	}
	else {
		// no pending/processed requests found
	}
}

#ifdef _CHECK_BB_SWAP
void UserManager::UpdateTrackingSafetyMeasure() {
	std::map<int, User*>::iterator it1;
	std::map<int, User*>::iterator it2;
	for (it1 = mFrameIDToUser.begin(); it1 != mFrameIDToUser.end(); it1++)
	{
		// reset status
		it1->second->SetTrackingIsSafe(true);

		// choose pair
		if (it1 != mFrameIDToUser.end()) {
			for (it2 = ++it1; it2 != mFrameIDToUser.end(); it2++) {
				cv::Rect r1 = it1->second->GetFaceBoundingBox();
				cv::Rect r2 = it2->second->GetFaceBoundingBox();
				// bbs intersect if area > 0
				bool intersect = ((r1 & r2).area() > 0);
				if (intersect) {
					// set safety status
					it1->second->SetTrackingIsSafe(false);
					it2->second->SetTrackingIsSafe(false);
				}
			}
		}
	}
}
#endif

// ----------------- helper functions

void UserManager::DrawUsers(cv::Mat &img)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		user::User* target_user = it->second;
		
		cv::Rect bb = target_user->GetFaceBoundingBox();

		// render user profile image
		cv::Mat profile_image;
		if (target_user->GetProfilePicture(profile_image))
		{
			cv::resize(profile_image, profile_image, cv::Size(100, 100));

			if(bb.y > 0 && bb.x < img.cols)
			{
				// check if roi overlapps image borders
				cv::Rect target_roi = cv::Rect(bb.x, bb.y - profile_image.rows, profile_image.cols, profile_image.rows);
				cv::Rect src_roi = cv::Rect(0, 0, profile_image.cols, profile_image.rows);

				if(target_roi.y < 0)
				{
					src_roi.y = -target_roi.y;
					src_roi.height += target_roi.y;
					target_roi.height += target_roi.y;
					target_roi.y = 0;
				}

				if (target_roi.x + profile_image.cols > img.cols)
				{
					src_roi.width = target_roi.x + profile_image.cols - img.cols;
					target_roi.width = target_roi.x + profile_image.cols - img.cols;
				}

				try {
					profile_image = profile_image(src_roi);
					profile_image.copyTo(img(target_roi));
				}
				catch (...) {
					// ...
				}

			}

			// render inside bb
			if(false)
			{
				cv::Rect picture_patch = cv::Rect(bb.x, bb.y, profile_image.cols, profile_image.rows);
				profile_image.copyTo(img(picture_patch));
			}
		}

		// draw identification status
		float font_size = 0.5;
		cv::Scalar color = cv::Scalar(0, 0, 0);
		std::string text1, text2;

		IdentificationStatus id_status;
		ActionStatus action;
		target_user->GetStatus(id_status, action);

		if (id_status == IDStatus_Identified)
		{
			int user_id = 0;
			std::string nice_name = "";
			target_user->GetUserID(user_id, nice_name);
			text1 = "Status: ID" + std::to_string(user_id);
			//text1 = "Status: " + nice_name + " - ID" + std::to_string(user_id);
			color = cv::Scalar(0, 255, 0);

			// confidence
			text1 += " Confidence: " + std::to_string(target_user->GetConfidence());
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

		cv::Scalar bg_color = cv::Scalar(0, 0, 0);
		if (!target_user->TrackingIsSafe()) {
			bg_color = cv::Scalar(0, 14, 88);
		}

		// draw flat background
		img(bg_patch) = bg_color;

		cv::putText(img, text1, cv::Point(bb.x+10, bb.y+20), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1, 8);
		cv::putText(img, text2, cv::Point(bb.x+10, bb.y+40), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1, 8);

		// draw face bounding box
		//cv::rectangle(img, bb, color, 2, cv::LINE_4);
	}
}
