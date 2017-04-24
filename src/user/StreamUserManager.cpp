#include <user\StreamUserManager.h>
#include <user\User.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tracking\FaceTracker.h>

// networking
#include <io/RequestHandler.h>
#include <io/RequestTypes.h>
#include <io/ResponseTypes.h>

using namespace  user;


// incorporate processed requests
void StreamUserManager::ProcessResponses()
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


			if (target_user->GetUserID() == response.mUserID)
			{
				// revalidation - might assign different user	
			}
			else
			{
				// Check for duplicate IDs
				for (auto its = mFrameIDToUser.begin(); its != mFrameIDToUser.end(); ++its)
				{
					// check if has assigned id - equals identified
					if ((duplicate_id = its->second->GetUserID()) > 0 && duplicate_id == response.mUserID)
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
			}


			if (duplicate_user)
			{
				std::cout << "-------------- DUPLICATE USER - RESETTING ID " << response.mUserID << std::endl;
			}
			else
			{
				// apply user identification
				target_user->SetUserID(response.mUserID, response.mUserNiceName);
				// prediction confidence
				target_user->mPredictionConfidence = response.mConfidence;
				target_user->mUserIDPredicted = response.mUserID;

				// remove all images form grid
				target_user->pGrid->Clear();


				// profile picture
				if (!response.mImage.empty())
				{
					target_user->SetProfilePicture(response.mImage);
				}
				else
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
	// Prediction Feedback	
	// ============================================= //
	io::PredictionFeedback pred_response;
	while (pRequestHandler->PopResponse(&pred_response, request_lookup, &req_type))
	{

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

			// update user prediction
			target_user->mUserIDPredicted = pred_response.mUserID;
			target_user->mPredictionConfidence = pred_response.mConfidence;
			std::cout << "Prediction: ID" << pred_response.mUserID << " | conf: " << pred_response.mConfidence << std::endl;

		}
		else {
			std::cout << "------ USER HAS LEFT SCENE. IGNORE THIS RESPONSE." << std::endl;
		}
	}


	// --------------------- MISC ------------------------------//


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
			if (req_type == io::NetworkRequest_ProfilePictureUpdate)
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
		std::cout << "--- Error response | RequestID (" << req_type << "): " << err_response.mMessage << std::endl;

		// locate user for which request was sent
		std::map<io::NetworkRequest*, User*>::iterator it = mRequestToUser.find(request_lookup);

		if (it != mRequestToUser.end()) {
			// extract user
			User* target_user = it->second;
			io::NetworkRequest* target_request = it->first;

			// error during profile picture update - user does not match
			if (req_type == io::NetworkRequest_ProfilePictureUpdate) {
				// performed during other actions (updates)
				target_user->SetPendingProfilePicture(false);
			}else
			{
				// reset action status
				target_user->SetStatus(ActionStatus_Idle);
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

void StreamUserManager::GenerateRequests(cv::Mat scene_rgb)
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
		// 1. IDENTIFICATION
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
					//if (target_user->pGrid->nr_images() > 9) {
				}

				// extract images
				std::vector<cv::Mat*> face_patches;
				std::vector<int> sample_weights;
				bool has_samples = target_user->pGrid->ExtractUnprocessedImageBatchWithTimeout(5, 6, face_patches, sample_weights);

				if (has_samples) {

					//cv::imshow("identification", imgproc::ImageProc::createOne(face_patches, 1, 10));
					//cv::waitKey(0);
					//cv::destroyAllWindows();

					// make new identification request
#ifdef _DLIB_PREALIGN
					io::PartialImageIdentificationAligned* new_request = new io::PartialImageIdentificationAligned(pServerConn, face_patches, sample_weights, target_user->GetTrackingID());
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
				}

				// reset grid
				target_user->pGrid->ResetIfFullOrStagnating(6, 5);

			}
		}
		// ============================================= //
		// 2. UPDATES
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

				}

				// extract images
				std::vector<cv::Mat*> face_patches;
				std::vector<int> sample_weights;
				bool has_samples = target_user->pGrid->ExtractUnprocessedImageBatchWithTimeout(5, 5, face_patches, sample_weights);

				if (has_samples) {
					// extract images
					int user_id = target_user->GetUserID();


					//cv::imshow("update", imgproc::ImageProc::createOne(face_patches, 1, 10));
					//cv::waitKey(0);
					//cv::destroyAllWindows();

#ifdef _DLIB_PREALIGN
					io::PartialUpdateAligned* new_request = new io::PartialUpdateAligned(pServerConn, face_patches, sample_weights, user_id);
#else
					throw;
#endif
					// make update request
					pRequestHandler->addRequest(new_request);

					// update linking
					mRequestToUser[new_request] = target_user;
					mUserToRequests[target_user].insert(new_request);

					// set user action status
					target_user->SetStatus(ActionStatus_Waiting);
					//target_user->pGrid->Clear();

				}

				// reset grid
				if (target_user->pGrid->ResetIfFullOrStagnating(10, 7)) {
					std::cout << "--- Grid resetted!\n";
				}
			}
		}
		// ============================================= //
		// 3. CLOSED SET TRACK VERIFICATION
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
					if (target_user->pGrid->HasEnoughOrFrontalPictures(3)) {
						// extract images
						std::vector<cv::Mat*> face_patches;
						std::vector<int> sample_weights;
						target_user->pGrid->ExtractGrid(face_patches, sample_weights);
						

#ifdef _CLOSED_SET_REVALIDATION
						// closed set re-validation
						std::unordered_set<int> target_set = target_user->mClosedSetConfusionIDs;
#else
						// forces open set re-validation (validation against unknown -1)
						std::unordered_set<int> target_set = {-1};
#endif


#ifdef _DLIB_PREALIGN
						io::ImageIdentificationAlignedCS* new_request = new io::ImageIdentificationAlignedCS(
							pServerConn, face_patches, target_set
						);
#else
						throw;
#endif
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

	}	// end user iteration
}