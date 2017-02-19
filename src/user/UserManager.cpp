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
	mpDlibAligner = new features::DlibFaceAligner();
	mpDlibAligner->Init();
#endif

	std::cout << "--- UserManager initialized" << std::endl;

	return true;
}

void UserManager::UpdateFaceData(std::vector<tracking::Face> faces, std::vector<int> user_ids) {
	for (size_t i = 0; i < faces.size(); i++) {

#ifdef _DEBUG_USERMANAGER
		if(mFrameIDToUser.count(user_ids[i]) == 0)
		{
			throw std::invalid_argument("Updating invalid User!");
		}
#endif

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
			mFrameIDToUser[scene_id] = new User(
#ifdef _DLIB_PREALIGN
				mpDlibAligner
#endif
			);
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

#ifdef _CHECK_TRACKING_CONF
	// update the tracking safety status
	UpdateTrackingStatus();
#endif

}

// incorporate processed requests: update user ids
void UserManager::ProcessResponses()
{
	io::NetworkRequest* request_lookup = nullptr;	// careful! the request corresponding to this pointer is already deleted!
	io::NetworkRequestType req_type;
	user::IdentificationStatus id_status;

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
					target_user->SetProfilePicture(response.mImage);
				}else
				{
					//std::cout << "------ no profile picture was taken before\n";
				}
				// update confidence
				target_user->SetConfidence(response.mConfidence);
			}

			// reset action status
			target_user->SetPendingProfilePicture(false);
			target_user->SetStatus(ActionStatus_Idle);
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
	// Update response (successful)
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
			target_user->SetStatus(ActionStatus_Idle);

			target_user->GetStatus(id_status);
			
			// if the update was robust and successful:
			if (
				(target_request->cRequestType == io::NetworkRequest_EmbeddingCollectionByIDRobust ||
				 target_request->cRequestType == io::NetworkRequest_EmbeddingCollectionByIDAlignedRobust
				)
				&&
				id_status == IDStatus_Uncertain
				) {
				// user has reidentified itself
				target_user->SetStatus(IDStatus_Identified);
			}
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
				target_user->SetProfilePicture(img_response.mImage);
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
			target_user->SetStatus(ActionStatus_Idle);

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
			if (
				req_type == io::NetworkRequest_ImageIdentification ||
				req_type == io::NetworkRequest_ImageIdentificationAligned
				) {
				target_user->SetStatus(user::IDStatus_Unknown);
				target_user->SetStatus(ActionStatus_Idle);
			}
			// error during update - not enough "good"/destinctive feature vectors (most vectors are around threshold)
			// trash update and start again
			else if (
				req_type == io::NetworkRequest_EmbeddingCollectionByID ||
				req_type == io::NetworkRequest_EmbeddingCollectionByIDAligned ||
				req_type == io::NetworkRequest_EmbeddingCollectionByIDAlignedRobust ||
				req_type == io::NetworkRequest_EmbeddingCollectionByName
					) 
			{
				target_user->SetStatus(ActionStatus_Idle);
			}
			// error during profile picture update - user does not match
			else if (req_type == io::NetworkRequest_ProfilePictureUpdate) {
				target_user->SetPendingProfilePicture(false);
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


		// do nothing
		if (action == ActionStatus_Waiting
			|| action == ActionStatus_WaitForCertainTracking) {
			continue;
		}

		// ============================================= //
		// 1. Unknown
		// ============================================= //
		if (id_status == IDStatus_Unknown)
		{
			// new user in scene
			if (action == ActionStatus_Idle) {
				target_user->SetStatus(ActionStatus_DataCollection);
				action = ActionStatus_DataCollection;
				// reset unsafe samples
				target_user->pGrid->Clear();
			}

			// collect images for identification
			if (action == ActionStatus_DataCollection) {

				if (target_user->TryToRecordFaceSample(scene_rgb))
				{
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
						pRequestHandler->addRequest(new_request, true);

						// update linking
						mRequestToUser[new_request] = target_user;
						mUserToRequests[target_user].insert(new_request);

						// set user action status
						target_user->SetPendingProfilePicture(true);	// might get it from server
						target_user->SetStatus(ActionStatus_Waiting);
						target_user->pGrid->Clear();
					}
				}
			}
		}
		// ============================================= //
		// 2. Uncertain
		// ============================================= //
		else if (id_status == IDStatus_Uncertain)
		{
			// if nothing to do: collect updates
			if (action == ActionStatus_Idle) {
				target_user->SetStatus(ActionStatus_DataCollection);
				action = ActionStatus_DataCollection;
				// reset unsafe samples
				target_user->pGrid->Clear();
			}

			// collect images for identification
			if (action == ActionStatus_DataCollection) {

				if (target_user->TryToRecordFaceSample(scene_rgb))
				{
					// if enough images, request identification
					if (target_user->pGrid->nr_images() > 9) {

						// extract images
						std::vector<cv::Mat*> face_patches = target_user->pGrid->ExtractGrid();

						// make new identification request
#ifdef _DLIB_PREALIGN
						io::EmbeddingCollectionByID* new_request = new io::EmbeddingCollectionByID(
							pServerConn, face_patches, target_user->GetUserID(),
							io::NetworkRequest_EmbeddingCollectionByIDAlignedRobust	// specified request type
						);
#else
						io::EmbeddingCollectionByID* new_request = new io::EmbeddingCollectionByID(
							pServerConn, face_patches, target_user->GetUserID(),
							io::NetworkRequest_EmbeddingCollectionByIDRobust	// specified request type
						);
#endif
						pRequestHandler->addRequest(new_request, true);

						// update linking
						mRequestToUser[new_request] = target_user;
						mUserToRequests[target_user].insert(new_request);

						// set user action status
						target_user->SetStatus(ActionStatus_Waiting);
						target_user->pGrid->Clear();
					}
				}
			}
		}
		// ============================================= //
		// 3. Identified
		// ============================================= //
		else if (id_status == IDStatus_Identified) {

			// if nothing to do: collect updates
			if (action == ActionStatus_Idle) {
				target_user->SetStatus(ActionStatus_DataCollection);
				action = ActionStatus_DataCollection;
			}

			// Update/assign profile picture
			if (target_user->NeedsProfilePicture() &&
				target_user->IsViewedFromFront())
			{
				target_user->SetPendingProfilePicture(true);
				cv::Mat profile_picture = scene_rgb(target_user->GetFaceBoundingBox());
				// scale
				cv::resize(profile_picture, profile_picture, cv::Size(120, 120));
				// make request
				io::ProfilePictureUpdate* new_request = new io::ProfilePictureUpdate(pServerConn, target_user->GetUserID(), profile_picture);
				pRequestHandler->addRequest(new_request, true);
				// update linking
				mRequestToUser[new_request] = target_user;
				mUserToRequests[target_user].insert(new_request);
			}


			// send model updates - reinforced learning
			if (action == ActionStatus_DataCollection) {

				if (target_user->TryToRecordFaceSample(scene_rgb))
				{

#ifdef FACEGRID_RECORDING

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
						new_request = new io::EmbeddingCollectionByID(
							pServerConn, face_patches, user_id,
							io::NetworkRequest_EmbeddingCollectionByIDAligned	// specified request type
						);
#else
						// standard update
						new_request = new io::EmbeddingCollectionByID(pServerConn, face_patches, user_id);
#endif
						pRequestHandler->addRequest(new_request);

						// update linking
						mRequestToUser[new_request] = target_user;
						mUserToRequests[target_user].insert(new_request);

						// set user action status
						target_user->SetStatus(ActionStatus_Waiting);
						target_user->pGrid->Clear();
					}
#endif
				}
			}



		}

	}
}



//// send identification requests for all unknown users
//void UserManager::GenerateRequests_DEPRECATED(cv::Mat scene_rgb)
//{
//	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
//	{
//		IdentificationStatus id_status;
//		ActionStatus action;
//		TrackingStatus tracking_status;
//		user::User* target_user = it->second;
//		target_user->GetStatus(id_status, action);
//		target_user->GetStatus(tracking_status);
//
//		//std::cout << "--- id_Status: "<< id_status << " | action: "<< action << std::endl;
//		// request user identification
//
//		if (id_status == IDStatus_Uncertain)
//		{
//			
//		}
//		else if (id_status == IDStatus_Unknown)
//		{
//
//			// do nothing
//			// 1. ID pending
//			if (action == ActionStatus_Waiting) {
//				continue;
//			}
//
//			// new user in scene
//			if (action == ActionStatus_Idle) {
//				target_user->SetStatus(ActionStatus_DataCollection);
//				action = ActionStatus_DataCollection;
//			}
//
//			// collect images for identification
//			if (action == ActionStatus_DataCollection) {
//
//#ifdef FACEGRID_RECORDING
//				// check if face should be recorded
//				tracking::Face face;
//				if (target_user->GetFaceData(face)) {
//
//					int roll, pitch, yaw;
//					face.GetEulerAngles(roll, pitch, yaw);
//					// face from this pose not yet recorded
//					if (target_user->pGrid->IsFree(roll, pitch, yaw)) {
//						cv::Rect2f facebb = target_user->GetFaceBoundingBox();
//						cv::Mat face_snap = scene_rgb(facebb);
//						cv::Mat aligned;
//#ifdef _DLIB_PREALIGN
//						if (mpDlibAligner->AlignImage(96, face_snap, aligned))
//#else
//						aligned = face_snap;
//						// resize (requests needs all squared with same size)
//						cv::resize(aligned, aligned, cv::Size(120, 120));
//#endif
//						{
//							try
//							{
//								target_user->pGrid->StoreSnapshot(roll, pitch, yaw, aligned);
//							}
//							catch (...)
//							{
//							}
//						}
//					}	// /free pose position
//
//					// if enough images, request identification
//					if (target_user->pGrid->nr_images() > 9) {
//
//						// extract images
//						std::vector<cv::Mat*> face_patches = target_user->pGrid->ExtractGrid();
//
//						// make new identification request
//#ifdef _DLIB_PREALIGN
//						io::ImageIdentificationAligned* new_request = new io::ImageIdentificationAligned(pServerConn, face_patches);
//#else
//						IDReq* new_request = new IDReq(pServerConn, face_patches);
//#endif
//						pRequestHandler->addRequest(new_request, true);
//
//						// update linking
//						mRequestToUser[new_request] = target_user;
//						mUserToRequests[target_user].insert(new_request);
//
//						// set user action status
//						target_user->SetPendingProfilePicture(true);	// might get it from server
//						target_user->SetStatus(ActionStatus_Waiting);
//						target_user->pGrid->Clear();
//					}
//				}
//#endif
//
//			}
//
//		}
//		else if (id_status == IDStatus_Identified) {
//			
//			if(tracking_status == user::TrackingStatus_Certain)
//			{
//				if (action == ActionStatus_WaitForCertainTracking)
//				{
//					// do identification request (e.g. robust update)
//
//					// block further requests
//					target_user->SetStatus(user::ActionStatus_Waiting);
//				}
//			}else
//			{
//				// wait till internal tracking status has changed
//				target_user->SetStatus(ActionStatus_WaitForCertainTracking);
//				continue;
//			}
//
//			// waiting (e.g. update pending)
//			if (action == ActionStatus_Waiting) {
//				continue;
//			}
//			// if nothing to do: collect updates
//			if (action == ActionStatus_Idle) {
//				target_user->SetStatus(ActionStatus_DataCollection);
//				action = ActionStatus_DataCollection;
//			}
//
//			// Update/assign profile picture
//			if (target_user->NeedsProfilePicture() &&
//				target_user->IsViewedFromFront())
//			{
//				target_user->SetPendingProfilePicture(true);
//				cv::Mat profile_picture = scene_rgb(target_user->GetFaceBoundingBox());
//				// scale
//				cv::resize(profile_picture, profile_picture, cv::Size(120, 120));
//				// make request
//				io::ProfilePictureUpdate* new_request = new io::ProfilePictureUpdate(pServerConn, target_user->GetUserID(), profile_picture);
//				pRequestHandler->addRequest(new_request, true);
//				// update linking
//				mRequestToUser[new_request] = target_user;
//				mUserToRequests[target_user].insert(new_request);
//			}
//
//			// send model updates - reinforced learning
//			if (action == ActionStatus_DataCollection) {
//#ifdef FACEGRID_RECORDING
//				// check if face should be recorded
//				tracking::Face face;
//				if (target_user->GetFaceData(face)) {
//
//					int roll, pitch, yaw;
//					face.GetEulerAngles(roll, pitch, yaw);
//
//					// face from this pose not yet recorded
//					if (target_user->pGrid->IsFree(roll, pitch, yaw)) {
//
//						cv::Rect2f facebb = target_user->GetFaceBoundingBox();
//
//						// collect another image
//						cv::Mat face_snap = scene_rgb(facebb);
//
//						// detect and warp face
//						cv::Mat aligned;
//						
//#ifdef _DLIB_PREALIGN
//							if (mpDlibAligner->AlignImage(96, face_snap, aligned)) 
//#else
//						aligned = face_snap;
//						// resize (requests needs all squared with same size)
//						cv::resize(aligned, aligned, cv::Size(120, 120));
//#endif
//							{
//								// save
//								try
//								{
//									// add face if not yet capture from this angle
//									target_user->pGrid->StoreSnapshot(roll, pitch, yaw, aligned);
//									//std::cout << "--- take snapshot: " << target_user->pGrid->nr_images() << std::endl;
//								}
//								catch (...)
//								{
//								}
//							}
//					}	// /free pose position
//
//						// if enough images, request identification
//					if (target_user->pGrid->nr_images() > 9) {
//
//						// extract images
//						std::vector<cv::Mat*> face_patches = target_user->pGrid->ExtractGrid();
//
//						if (face_patches[0]->cols == 0)
//						{
//							std::cout << "----------- WHY? -----------" << std::endl;
//						}
//						int user_id; std::string user_name;
//
//						// TODO: DEBUG HERE
//						// ID -1
//						target_user->GetUserID(user_id, user_name);
//
//						io::EmbeddingCollectionByID* new_request;
//#ifdef _DLIB_PREALIGN
//#ifdef _CHECK_TRACKING_CONF
//
//						user::TrackingStatus tracking_status;
//						target_user->GetStatus(tracking_status);
//						// robust update: check update for model consistency
//						if (tracking_status == user::TrackingStatus_Uncertain) {
//						new_request = new io::EmbeddingCollectionByID(
//							pServerConn, face_patches, user_id,
//							io::NetworkRequest_EmbeddingCollectionByIDAlignedRobust	// specified request type
//						);
//					}else
//#endif
//					
//						{new_request = new io::EmbeddingCollectionByID(
//							pServerConn, face_patches, user_id, 
//							io::NetworkRequest_EmbeddingCollectionByIDAligned	// specified request type
//						);}
//
//#else
//						// standard update
//						new_request = new io::EmbeddingCollectionByID(pServerConn, face_patches, user_id);
//#endif
//						pRequestHandler->addRequest(new_request);
//
//						// update linking
//						mRequestToUser[new_request] = target_user;
//						mUserToRequests[target_user].insert(new_request);
//
//						// set user action status
//						target_user->SetStatus(ActionStatus_Waiting);
//						target_user->pGrid->Clear();
//					}
//				}	//	/end face data available
//#endif
//
//			}
//
//
//
//		}
//		
//	}
//}

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

#ifdef _CHECK_TRACKING_CONF
void UserManager::UpdateTrackingStatus() {

	std::map<int, bool> scene_ids_uncertain;

	// get current tracking status
	for (auto uit1 = mFrameIDToUser.begin(); uit1 != mFrameIDToUser.end(); ++uit1)
	{
		// choose pair (if not last element)
		if (uit1 != std::prev(mFrameIDToUser.end())) {
			for (auto it2 = std::next(uit1); it2 != mFrameIDToUser.end(); ++it2) {
				cv::Rect r1 = uit1->second->GetFaceBoundingBox();
				cv::Rect r2 = it2->second->GetFaceBoundingBox();

				// bbs intersect if area > 0
				bool intersect = ((r1 & r2).area() > 0);
				if (intersect) {
					// track scene ids
					scene_ids_uncertain[uit1->first] = true;
					scene_ids_uncertain[it2->first] = true;
				}
			}
		}
	}

	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		IdentificationStatus s;
		ActionStatus as;
		it->second->GetStatus(s);
		it->second->GetStatus(as);

		// currently unsafe tracking
		if(scene_ids_uncertain.count(it->first) > 0)
		{
			// update temp tracking status
			it->second->SetStatus(TrackingStatus_Uncertain);


			// tracking is uncertain atm
			if(s==IDStatus_Identified)
			{
				// update id status
				it->second->SetStatus(IDStatus_Uncertain);
				// cancel all pending requests
				// atm: no robust request besides reidentification
				// todo: else cancel all other requests here (prevent delayed reidentification at an unsafe state)

				// wait till tracking is save again, then do reidentification
				it->second->SetStatus(ActionStatus_WaitForCertainTracking);
			}
			else if(s == IDStatus_Uncertain) {


				// cancel possible pending reidentification requests
				if (as == ActionStatus_Waiting) {
					CancelAndDropAllUserRequests(it->second);
				}

				// wait till tracking is save again, then do reidentification
				it->second->SetStatus(ActionStatus_WaitForCertainTracking);

			}
			else if (s == IDStatus_Unknown) {
				// cancel data collection - unambiguous
				if (as == ActionStatus_Waiting) {
					CancelAndDropAllUserRequests(it->second);
				}

				it->second->SetStatus(ActionStatus_WaitForCertainTracking);
			}
		// safe tracking
		}else
		{
			// update temp tracking status
			it->second->SetStatus(TrackingStatus_Certain);

			// tracking safe again - reset samples and start data collection
			if(s==IDStatus_Uncertain && as==ActionStatus_WaitForCertainTracking)
			{
				it->second->SetStatus(ActionStatus_Idle);
			}
			else if (s == IDStatus_Unknown && as == ActionStatus_WaitForCertainTracking) {
				it->second->SetStatus(ActionStatus_Idle);
			}
		}

	}


}
#endif

//#ifdef _CHECK_TRACKING_CONF
//void UserManager::UpdateTrackingStatus() {
//
//
//	for (auto uit1 = mFrameIDToUser.begin(); uit1 != mFrameIDToUser.end(); ++uit1)
//	{
//		// reset status
//		uit1->second->SetStatus(user::TrackingStatus_Certain);
//	}
//	// temporary status: uncertain
//	for (auto uit1 = mFrameIDToUser.begin(); uit1 != mFrameIDToUser.end(); ++uit1)
//	{
//		// choose pair (if not last element)
//		if (uit1 != std::prev(mFrameIDToUser.end())) {
//			for (auto it2 = std::next(uit1); it2 != mFrameIDToUser.end(); ++it2) {
//				cv::Rect r1 = uit1->second->GetFaceBoundingBox();
//				cv::Rect r2 = it2->second->GetFaceBoundingBox();
//
//				// bbs intersect if area > 0
//				bool intersect = ((r1 & r2).area() > 0);
//				if (intersect) {
//					// set safety status
//					uit1->second->SetStatus(user::TrackingStatus_Uncertain);
//					it2->second->SetStatus(user::TrackingStatus_Uncertain);
//				}
//
//			}
//		}
//	}
//
//}
//#endif

// ----------------- API functions

std::vector<std::pair<int, int>> UserManager::GetUserandTrackingID() {
	std::vector<std::pair<int, int>> out;
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		out.push_back(std::make_pair(it->second->GetUserID(), it->first));
	}
	return out;
}

std::vector<std::pair<int, cv::Mat>> UserManager::GetSceneProfilePictures() {
	std::vector<std::pair<int, cv::Mat>> out;
	cv::Mat profile_pic;
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		if (it->second->GetProfilePicture(profile_pic)) {
			out.push_back(std::make_pair(it->second->GetUserID(), profile_pic));
		}
	}
	return out;
}

std::vector<std::pair<int, cv::Mat>> UserManager::GetAllProfilePictures() {
	std::vector<std::pair<int, cv::Mat>> out;
	// request images from server
	io::GetProfilePictures req(pServerConn);
	pServerConn->Connect();
	req.SubmitRequest();
	// wait for reponse
	io::ProfilePictures resp(pServerConn);
	int response_code = 0;
	if (!resp.Load(&response_code)) {
		// error
	}
	else {
#ifdef _DEBUG_USERMANAGER
		if (resp.mUserIDs.size() != resp.mUserIDs.size()) {
			std::cout << "--- Error: size(user_ids) != size(user_profile_pictures)!\n";
		}
#endif
		// load images
		for (size_t i = 0; i < resp.mUserIDs.size(); i++) {
			out.push_back(std::make_pair(resp.mUserIDs[i], resp.mImages[i]));
		}
	}
	pServerConn->Close();
	return out;
}

bool UserManager::GetUserID(const cv::Mat &face_capture, int &user_id) {
	std::vector<cv::Mat> face_patches = { face_capture };
	IDReq id_request(pServerConn, face_patches);
	pServerConn->Connect();
	id_request.SubmitRequest();
	user_id = -1;
	bool succ = false;
	// wait for reponse
	io::IdentificationResponse id_response(pServerConn);
	int response_code = 0;
	if (!id_response.Load(&response_code)) {
		//std::cout << "--- An error occurred during update: ResponseType " << response_code << " | expected: " << id_response.cTypeID << std::endl;
	}
	else {
		succ = true;
		user_id = id_response.mUserID;
	}
	pServerConn->Close();
	return succ;
}

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

		if (id_status == IDStatus_Identified || id_status == IDStatus_Uncertain)
		{
			int user_id = 0;
			std::string nice_name = "";
			target_user->GetUserID(user_id, nice_name);
			text1 = "Status: ID" + std::to_string(user_id);
			//text1 = "Status: " + nice_name + " - ID" + std::to_string(user_id);
			color = cv::Scalar(0, 255, 0);

			// confidence
			//text1 += " Confidence: " + std::to_string(target_user->GetConfidence());

			if (id_status == IDStatus_Uncertain) {
				if (action == ActionStatus_WaitForCertainTracking) {
					text1 += " | waiting for safe tracking";
				}
				else if (action == ActionStatus_Waiting) {
					text1 += " | pending reidentification";
				}
				else if (action == ActionStatus_DataCollection) {
					text1 += " | sampling (" + std::to_string(target_user->pGrid->nr_images()) + ")";
				}
			}

			
			
		}
		else
		{
			text1 = "Status: unknown";
			if (action == ActionStatus_DataCollection) {
				text2 = "Sampling ("+std::to_string(target_user->pGrid->nr_images())+")";
			}
			else if (action == ActionStatus_Waiting) {
				text2 = "ID pending";
			}
			else if (action == ActionStatus_Idle) {
				text2 = "Idle";
			}
			else if (action == ActionStatus_WaitForCertainTracking) {
				text2 = "Wait for safe tracking";
			}
			color = cv::Scalar(0, 0, 255);
			
		}

		int baseline = 0;
		cv::Size textSize = cv::getTextSize(text1, cv::FONT_HERSHEY_SIMPLEX, font_size, 1, &baseline);
		cv::Rect bg_patch = cv::Rect(bb.x, bb.y, textSize.width + 20, textSize.height + 15);

		cv::Scalar bg_color = cv::Scalar(0, 0, 0);

		user::TrackingStatus tracking_status;
		target_user->GetStatus(tracking_status);
		if (tracking_status == user::TrackingStatus_Uncertain) {
			bg_color = cv::Scalar(0, 14, 88);
		}

		// draw flat background
		img(bg_patch) = bg_color;

		cv::putText(img, text1, cv::Point(bb.x+10, bb.y+20), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1, 8);
		cv::putText(img, text2, cv::Point(bb.x+10, bb.y+40), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1, 8);

		// draw face bounding box
		if (id_status == IDStatus_Uncertain) {
			cv::rectangle(img, bb, cv::Scalar(0, 14, 88), 2, cv::LINE_4);
		}
	}
}
