#include <user\BaseUserManager.h>
#include <user\User.h>
#include <opencv2/imgproc.hpp>
#include <io/RequestHandler.h>
#include <io/RequestTypes.h>
#include <io/ResponseTypes.h>
#include <opencv2/highgui/highgui.hpp>
#include <tracking\FaceTracker.h>
#include <gui/GUI.h>
#include <gui/UserView.h>

using namespace  user;

bool BaseUserManager::Init(io::TCPClient* connection, io::NetworkRequestHandler* handler)
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

	std::cout << "--- BaseUserManager initialized" << std::endl;

	return true;
}

void BaseUserManager::UpdateFaceData(std::vector<tracking::Face> faces, std::vector<int> user_ids) {
	for (size_t i = 0; i < faces.size(); i++) {

#ifdef _DEBUG_BaseUserManager
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
void BaseUserManager::RefreshUserTracking(
	const std::vector<int> &user_scene_ids, 
	std::vector<cv::Rect2f> bounding_boxes,
	std::vector<cv::Point3f> positions
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
				mpDlibAligner ,
#endif
				scene_id
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
			target_user->UpdateFaceBoundingBox(bounding_boxes[user_index]);
			// update position in 3D
			target_user->SetPosition3D(positions[user_index]);
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

			// drop all collected identification samples
			IdentificationStatus id_status;
			target_user->GetStatus(id_status);
			if(id_status == IDStatus_Unknown)
			{
				io::DropIdentificationSamples* new_request = new io::DropIdentificationSamples(pServerConn, user_frame_id);
				pRequestHandler->addRequest(new_request, true);
			}

			// user has left scene - delete tracking instance
			delete(target_user);

#ifdef _DEBUG_BaseUserManager
			std::cout << "=== User has left scene - removing UserSceneID " << it->first << std::endl;
#endif

			// remove mapping
			mFrameIDToUser.erase(it++);	// increment after deletion
		}
	}

}

void BaseUserManager::CancelAndDropAllUserRequests(User* user) {
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


void BaseUserManager::UpdateTrackingStatus() {

	IdentificationStatus ids;

	std::map<int, bool> scene_ids_uncertain;

	// check human user tracking status
	for (auto uit1 = mFrameIDToUser.begin(); uit1 != mFrameIDToUser.end(); ++uit1)
	{

		// ------------------------ check human status


		uit1->second->GetStatus(ids);

		// update face detection counter
		uit1->second->IncrementFaceDetectionStatus();

		// increment bounding box movement status
		uit1->second->IncrementBBMovementStatus();

		// update overall status
		if (ids == IDStatus_IsObject) {
			if (!uit1->second->IsTrackingObject()) {
				uit1->second->SetStatus(IDStatus_Unknown);
			}
		}
		else {
			if (uit1->second->IsTrackingObject()) {
				// reset UserID
				uit1->second->ResetUserIdentity();
				// set to object
				uit1->second->SetStatus(IDStatus_IsObject);
			}
		}

		// ------------------------ check tracker safety
#ifdef _CHECK_TRACKING_CONF
		// choose pair (if not last element)
		if (uit1 != std::prev(mFrameIDToUser.end())) {
			for (auto it2 = std::next(uit1); it2 != mFrameIDToUser.end(); ++it2) {

				// check position
				cv::Point3f p1 = uit1->second->GetPosition3D();
				cv::Point3f p2 = it2->second->GetPosition3D();

				// if users closer than 0.8 m
				if (abs(p1.z - p2.z)<0.8) {
					cv::Rect r1 = uit1->second->GetFaceBoundingBox();
					cv::Rect r2 = it2->second->GetFaceBoundingBox();

					// expand height
					r1.height = std::min(r1.height, 200);
					r2.height = std::min(r2.height, 200);

					// bbs intersect if area > 0
					bool intersect = ((r1 & r2).area() > 0);
					if (intersect) {
						// track scene ids
						scene_ids_uncertain[uit1->first] = true;
						scene_ids_uncertain[it2->first] = true;

						// update possible user confusions
						uit1->second->mClosedSetConfusionIDs.insert(it2->second->GetUserID());
						it2->second->mClosedSetConfusionIDs.insert(uit1->second->GetUserID());
					}
				}
			}
		}
#endif
	}

	// ------------------check tracking confidence
#ifdef _CHECK_TRACKING_CONF


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
			it->second->SetStatus(TrackingConsistency_Uncertain);

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
			it->second->SetStatus(TrackingConsistency_OK);

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
#endif

}


// ----------------- API functions

std::vector<std::pair<int, int>> BaseUserManager::GetUserandTrackingID() {
	std::vector<std::pair<int, int>> out;
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		out.push_back(std::make_pair(it->second->GetUserID(), it->first));
	}
	return out;
}

std::vector<std::pair<int, cv::Mat>> BaseUserManager::GetSceneProfilePictures() {
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

std::vector<std::pair<int, cv::Mat>> BaseUserManager::GetAllProfilePictures() {
	std::vector<std::pair<int, cv::Mat>> out;
	// request images from server
	io::GetProfilePictures req(pServerConn);

	// submit priority request
	pRequestHandler->addRequest(&req, true);

	// wait for response (blocking)
	io::NetworkRequest* request_lookup = nullptr;	// careful! the request corresponding to this pointer is already deleted!
	user::IdentificationStatus id_status;

	io::ProfilePictures response;

	while (!pRequestHandler->PopResponse(&response, request_lookup))
	{
		// breaks when response is received
	}

	// load images
	for (size_t i = 0; i < response.mUserIDs.size(); i++) {
		std::cout << "here\n";
		out.push_back(std::make_pair(response.mUserIDs[i], response.mImages[i]));
	}

	return out;
}

//std::vector<std::pair<int, cv::Mat>> BaseUserManager::GetAllProfilePictures() {
//	std::vector<std::pair<int, cv::Mat>> out;
//	// request images from server
//	io::GetProfilePictures req(pServerConn);
//#ifndef _KEEP_SERVER_CONNECTION
//	pServerConn->Connect();
//#endif
//	req.SubmitRequest();
//	// wait for reponse
//	io::ProfilePictures resp(pServerConn);
//	int response_code = 0;
//	if (!resp.Load(&response_code)) {
//		// error
//	}
//	else {
//#ifdef _DEBUG_BaseUserManager
//		if (resp.mUserIDs.size() != resp.mUserIDs.size()) {
//			std::cout << "--- Error: size(user_ids) != size(user_profile_pictures)!\n";
//		}
//#endif
//		// load images
//		for (size_t i = 0; i < resp.mUserIDs.size(); i++) {
//			out.push_back(std::make_pair(resp.mUserIDs[i], resp.mImages[i]));
//		}
//	}
//#ifndef _KEEP_SERVER_CONNECTION
//	pServerConn->Close();
//#endif
//	return out;
//}

void BaseUserManager::GetAllProfilePictures(std::vector<cv::Mat> &pictures, std::vector<int> &user_ids) {

	// request images from server
	io::GetProfilePictures req(pServerConn);

	// submit priority request
	pRequestHandler->addRequest(&req);

	// wait for response (blocking)
	io::NetworkRequest* request_lookup = nullptr;	// careful! the request corresponding to this pointer is already deleted!
	user::IdentificationStatus id_status;

	io::ProfilePictures response;

	while (!pRequestHandler->PopResponse(&response, request_lookup))
	{
		// breaks when response is received
	}

	std::cout << "test" << std::endl;
	// load images
	for (size_t i = 0; i < response.mUserIDs.size(); i++) {
		std::cout << "here\n";
		// load images
		for (size_t i = 0; i < response.mUserIDs.size(); i++) {
			user_ids.push_back(response.mUserIDs[i]);
			pictures.push_back(response.mImages[i]);
		}
	}

}

bool BaseUserManager::GetUserID(const cv::Mat &face_capture, int &user_id) {
	std::vector<cv::Mat> face_patches = { face_capture };
	IDReq id_request(pServerConn, face_patches);
#ifndef _KEEP_SERVER_CONNECTION
	pServerConn->Connect();
#endif
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
#ifndef _KEEP_SERVER_CONNECTION
	pServerConn->Close();
#endif
	return succ;
}

// ----------------- helper functions

void BaseUserManager::RenderGUI(cv::Mat &img)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		user::User* target_user = it->second;
		
		cv::Rect bb = target_user->GetFaceBoundingBox();

		// render user profile image
		cv::Mat profile_image;
		if (target_user->GetProfilePicture(profile_image))
		{
			gui::safe_copyTo(img, profile_image, cv::Rect(bb.x, bb.y - 100, 100, 100));

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
		cv::Scalar bg_color = cv::Scalar(0, 0, 0);
		std::string text1, text2;

		IdentificationStatus id_status;
		ActionStatus action;
		target_user->GetStatus(id_status, action);

		// show current prediction
		cv::putText(img, "Prediction: "+std::to_string(target_user->mUserIDPredicted)+" || " + std::to_string(target_user->mPredictionConfidence) + "/100", cv::Point(bb.x + 10, bb.y + 70), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1, 8);

		if (id_status == IDStatus_Identified || id_status == IDStatus_Uncertain)
		{
			int user_id = 0;
			std::string nice_name = "";
			target_user->GetUserID(user_id, nice_name);
			text1 = "Status: ID" + std::to_string(user_id);
			//text1 = "Status: " + nice_name + " - ID" + std::to_string(user_id);
			color = cv::Scalar(255, 255, 255);

			// unsafe tracking
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

				text1 += " | ID confusion : ";
				for (auto conf_id : target_user->mClosedSetConfusionIDs) {
					text1 += " " + std::to_string(conf_id);
				}
			}

			// ID dependent background
			if (user_id==1) {
				// red
				bg_color = cv::Scalar(0, 0, 255);
			}
			else if (user_id==2) {
				// green
				bg_color = cv::Scalar(50, 205, 50);
			}
			else if (user_id==3) {
				// blue
				bg_color = cv::Scalar(255, 0, 0);
			}
			else if (user_id==4) {
				// orange
				bg_color = cv::Scalar(0, 140, 255);
			}
			else if (user_id==5) {
				// light blue
				bg_color = cv::Scalar(170, 178, 32);
			}
			else if (user_id==6) {
				// violett
				bg_color = cv::Scalar(133, 21, 199);
			} 

			

		}
		else if (id_status == IDStatus_IsObject) {
			text1 = "OBJECT TRACKING";
			color = cv::Scalar(0, 0, 0);
			bg_color = cv::Scalar(0, 0, 255);
			// dont draw objects
			continue;
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

			text2 += " " + target_user->GetHumanStatusString();

			color = cv::Scalar(0, 0, 255);
		}

		int baseline = 0;
		cv::Size textSize = cv::getTextSize(text1, cv::FONT_HERSHEY_SIMPLEX, font_size, 1, &baseline);
		cv::Rect bg_patch = cv::Rect(bb.x, bb.y, textSize.width + 20, textSize.height + 15);

		user::TrackingConsistency tracking_status;
		target_user->GetStatus(tracking_status);
		if (tracking_status == user::TrackingConsistency_Uncertain) {
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
