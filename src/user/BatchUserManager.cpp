#include <user\BatchUserManager.h>
#include <user\User.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tracking\FaceTracker.h>

// networking
#include <io/RequestHandler.h>
#include <io/RequestTypes.h>
#include <io/ResponseTypes.h>

using namespace  user;


// incorporate processed requests: update user ids
void BatchUserManager::ProcessResponses()
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
#ifdef _DEBUG_BatchUserManager
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
					its->second->ResetUserIdentity();
					target_user->ResetUserIdentity();
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
			target_user->ResetUserIdentity();
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
			// reset action status
			target_user->SetStatus(ActionStatus_Idle);

			target_user->GetStatus(id_status);
			
			// if the update was robust and successful:
			if (
				(target_request->cRequestType == io::NetworkRequest_EmbeddingCollectionByIDRobust ||
				 target_request->cRequestType == io::NetworkRequest_EmbeddingCollectionByIDAlignedRobust
				)
				//&& id_status == IDStatus_Uncertain
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
				req_type == io::NetworkRequest_EmbeddingCollectionByIDRobust ||
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
void BatchUserManager::GenerateRequests(cv::Mat scene_rgb)
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
		// 2. Uncertain (if _CHECK_TRACKING_CONF is enabled)
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

// ------ closed set
#ifdef _CLOSED_SET_REVALIDATION
		// TODO: custom request

// ------ open set
#else

#endif

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
				target_user->IsViewedFromFront()
				//&& target_user->LooksPhotogenic()
				)
			{
				target_user->SetPendingProfilePicture(true);
				cv::Mat profile_picture = scene_rgb(target_user->GetFaceBoundingBox());
				// scale profile picture
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

						// TODO: DEBUG HERE - fixed atm
						// ID -1
						target_user->GetUserID(user_id, user_name);

						io::EmbeddingCollectionByID* new_request;
#ifdef _DLIB_PREALIGN
	#ifdef _ALWAYS_DO_SAFE_UPDATE
						new_request = new io::EmbeddingCollectionByID(
							pServerConn, face_patches, user_id,
							io::NetworkRequest_EmbeddingCollectionByIDAlignedRobust	// specified request type
						);
	#else
						new_request = new io::EmbeddingCollectionByID(
							pServerConn, face_patches, user_id,
							io::NetworkRequest_EmbeddingCollectionByIDAligned	// specified request type
						);
	#endif

#else
	#ifdef _ALWAYS_DO_SAFE_UPDATE
						new_request = new io::EmbeddingCollectionByID(
							pServerConn, face_patches, user_id,
							io::NetworkRequest_EmbeddingCollectionByIDRobust	// specified request type
						);
	#else
						new_request = new io::EmbeddingCollectionByID(pServerConn, face_patches, user_id);
	#endif
#endif
						pRequestHandler->addRequest(new_request);

						// update linking
						mRequestToUser[new_request] = target_user;
						mUserToRequests[target_user].insert(new_request);

						// set user action status
						target_user->SetStatus(ActionStatus_Waiting);
						target_user->pGrid->Clear();
					}
#endif	// /FACEGRID_RECORDING 
				}
			}



		}

	}
}

