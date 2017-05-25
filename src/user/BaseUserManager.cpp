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
				scene_id,
				IDStatus_NonTarget	// init as object - till face is detected
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
		else if (ids == IDStatus_NonTarget) {
			// start target tracking if face has been seen once
			if (uit1->second->IsHuman()) {
				uit1->second->SetStatus(IDStatus_Unknown);
			}
		}else {
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
	io::GetProfilePictures *req = new io::GetProfilePictures(pServerConn);

#ifndef _KEEP_SERVER_CONNECTION
	pServerConn->Connect();
#endif

	// submit priority request
	pRequestHandler->addRequest(req, true);

	// wait for response (blocking)
	io::NetworkRequest* request_lookup = nullptr;	// careful! the request corresponding to this pointer is already deleted!
	user::IdentificationStatus id_status;

	// loop till we get response
	io::ProfilePictures response;
	int count = 0;
	while (!pRequestHandler->PopResponse(&response, request_lookup))
	{
		// breaks when response is received
		Sleep(10); // wait 10ms
		++count;
	}

	// load images
	for (size_t i = 0; i < response.mUserIDs.size(); i++) {
		out.push_back(std::make_pair(response.mUserIDs[i], response.mImages[i]));
	}

	return out;
}

bool BaseUserManager::GetAllProfilePictures(std::vector<cv::Mat> &pictures, std::vector<int> &user_ids) {

	// request images from server
	io::GetProfilePictures *req = new io::GetProfilePictures(pServerConn);

#ifndef _KEEP_SERVER_CONNECTION
	pServerConn->Connect();
#endif

	// submit priority request
	pRequestHandler->addRequest(req, true);

	// wait for response (blocking)
	io::NetworkRequest* request_lookup = nullptr;	// careful! the request corresponding to this pointer is already deleted!
	user::IdentificationStatus id_status;

	// loop till we get response
	io::ProfilePictures response;
	int count = 0;
	while (!pRequestHandler->PopResponse(&response, request_lookup))
	{
		// breaks when response is received
		Sleep(10); // wait 10ms
		++count;
		if (count > 200)
		{
			return false;
		}
	}

#ifndef _KEEP_SERVER_CONNECTION
	pServerConn->Close();
#endif

	// load images
	for (size_t i = 0; i < response.mUserIDs.size(); i++) {
		user_ids.push_back(response.mUserIDs[i]);
		pictures.push_back(response.mImages[i]);
	}

	return true;

}

bool BaseUserManager::GetUserID(const cv::Mat &face_capture, int &user_id) {
	std::vector<cv::Mat> face_patches = { face_capture };
	IDReq *id_request = new IDReq(pServerConn, face_patches);
	io::IdentificationResponse id_response(pServerConn);

	// receive response (blocking)
	if (pRequestHandler->SubmitAndWaitForSpecificResponse(id_request, &id_response)) {
		user_id = id_response.mUserID;
	}
	else {
		std::cout << "An error occured..." << std::endl;
		return false;
	}

	return true;
}

cv::Scalar BaseUserManager::GetUserColor(int user_id)
{
	// ID dependent background
	if (user_id == -1) {
		// gray
		return cv::Scalar(128, 128, 128);
	}else if (user_id == 1) {
		// red
		return cv::Scalar(0, 0, 255);
	}
	else if (user_id == 2) {
		// green
		return cv::Scalar(50, 205, 50);
	}
	else if (user_id == 3) {
		// blue
		return cv::Scalar(255, 0, 0);
	}
	else if (user_id == 4) {
		// orange
		return cv::Scalar(0, 140, 255);
	}
	else if (user_id == 5) {
		// light blue
		return cv::Scalar(170, 178, 32);
	}
	else if (user_id == 6) {
		// violett
		return cv::Scalar(133, 21, 199);
	}else if (user_id == 7) {
		// pink
		return cv::Scalar(255, 51, 255);
	}
	else if (user_id == 8) {
		// lachs
		return cv::Scalar(255, 153, 153);
	}
	else if (user_id == 9) {
		// lachs
		return cv::Scalar(204, 255, 153);
	}
	else if (user_id == 10) {
		// lachs
		return cv::Scalar(0, 102, 51);
	}
	else{
		// pink
		return cv::Scalar(255, 255, 51);
	}
}

// ----------------- helper functions

void BaseUserManager::RenderGUI(cv::Mat &img)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		user::User* target_user = it->second;

		IdentificationStatus id_status;
		ActionStatus action;
		target_user->GetStatus(id_status, action);

		// skip object tracking and non-target users
		if (
			id_status == IDStatus_IsObject
			//|| id_status == IDStatus_NonTarget
			)
		{
			continue;
		}

		// ============================================= //
		// 1. Profile picture
		// ============================================= //
		cv::Rect bb = target_user->GetFaceBoundingBox();
		cv::Mat profile_image;
		if (target_user->GetProfilePicture(profile_image))
		{
			gui::safe_copyTo(img, profile_image, cv::Rect(bb.x, bb.y - 108, 108, 108));
			// render inside bb
			if(false)
			{
				cv::Rect picture_patch = cv::Rect(bb.x, bb.y, profile_image.cols, profile_image.rows);
				profile_image.copyTo(img(picture_patch));
			}
		}

		// ============================================= //
		// 1.1 Face Bounding box
		// ============================================= //

		tracking::Face f;
		if(target_user->GetFaceData(f))
		{

			cv::Rect2f box = f.boundingBox;
			cv::Scalar bb_color = cv::Scalar(0, 0, 255);

			if (id_status == IDStatus_Identified)
			{
				bb_color = cv::Scalar(0, 255, 0);
			}

			try {
			// top left
			cv::line(img, cv::Point(box.x, box.y), cv::Point(box.x, box.y+10), bb_color, 2, 8);
			cv::line(img, cv::Point(box.x, box.y), cv::Point(box.x + 10, box.y), bb_color, 2, 8);
			// top right
			cv::line(img, cv::Point(box.x + box.width, box.y), cv::Point(box.x + box.width, box.y + 10), bb_color, 2, 8);
			cv::line(img, cv::Point(box.x + box.width, box.y), cv::Point(box.x + box.width - 10, box.y), bb_color, 2, 8);
			// bottom left
			cv::line(img, cv::Point(box.x, box.y + box.height), cv::Point(box.x, box.y + box.height -10), bb_color, 2, 8);
			cv::line(img, cv::Point(box.x, box.y + box.height), cv::Point(box.x + 10, box.y + box.height), bb_color, 2, 8);
			// bottom right
			cv::line(img, cv::Point(box.x + box.width, box.y + box.height), cv::Point(box.x + box.width, box.y + box.height - 10), bb_color, 2, 8);
			cv::line(img, cv::Point(box.x + box.width, box.y + box.height), cv::Point(box.x + box.width - 10, box.y + box.height), bb_color, 2, 8);
			}
			catch (...) {
				// ...
			}
		}

		// draw identification status
		float font_size = 0.5;
		float font_size_small = 0.3;
		cv::Scalar color = cv::Scalar(255, 255, 255);
		cv::Scalar bg_color = cv::Scalar(0, 0, 0);
		std::string text1, text2;

		// ============================================= //
		// 2. Text 1 row - ID
		// ============================================= //

		if (id_status == IDStatus_Identified || id_status == IDStatus_Uncertain)
		{
			int user_id = target_user->GetUserID();
			text1 = "ID " + std::to_string(user_id);

			// draw flat background
			cv::Rect bg_patch = cv::Rect(bb.x, bb.y, 108, 30);
			try {
				img(bg_patch) = GetUserColor(user_id);
			}
			catch (...) {
				// ...
			}

			// user id
			cv::putText(img, text1, cv::Point(bb.x + 10, bb.y + 20), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1.5, 8);
			// prediction confidence
			cv::putText(img, std::to_string(target_user->mPredictionConfidence) + "%", cv::Point(bb.x + 60, bb.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, 8);
		}
		else if(id_status == IDStatus_Unknown)
		{
			// use blurred out background
			cv::Rect bg_patch = cv::Rect(bb.x, bb.y, 108, 30);

			if(target_user->mUserIDPredicted != 0)
			{
				try {
					cv::Mat roi = img(bg_patch);
					cv::Mat color(bg_patch.size(), CV_8UC3, GetUserColor(target_user->mUserIDPredicted));
					double alpha = 0.5;
					cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
				}
				catch (...) {
					// ...
				}

				// show current prediction
				cv::putText(img, "ID " + std::to_string(target_user->mUserIDPredicted), cv::Point(bb.x + 10, bb.y + 20), cv::FONT_HERSHEY_SIMPLEX, font_size, color, 1.5, 8);
				cv::putText(img, std::to_string(target_user->mPredictionConfidence) + "%", cv::Point(bb.x + 60, bb.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, 8);

			}else
			{
				try {
					img(bg_patch) = cv::Scalar(0, 0, 0);
				}
				catch (...) {
					// ...
				}

				cv::putText(img, "ID pending", cv::Point(bb.x + 10, bb.y + 20), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(255, 255, 255), 1.5, 8);
			}
		}


		// ============================================= //
		// 3. Text 2 row - action status
		// ============================================= //

		// draw background
		if (
			id_status != IDStatus_NonTarget
			&& id_status != IDStatus_Unknown
			)
		{
			try {
				cv::Rect bg_patch = cv::Rect(bb.x, bb.y + 30, 108, 20);
				cv::Mat roi = img(bg_patch);
				cv::Mat color(bg_patch.size(), CV_8UC3, cv::Scalar(0, 0, 0));
				double alpha = 0.5;
				cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
			}
			catch (...) {
				// ...
			}
		}

		// print status
		cv::putText(img, target_user->GetActionStatusString(), cv::Point(bb.x + 10, bb.y + 40 + 2), cv::FONT_HERSHEY_SIMPLEX, font_size_small, cv::Scalar(255, 255, 255), 1.5, 8);

		// ============================================= //
		// 4. Text 3 row - progress
		// ============================================= //

		if (id_status == IDStatus_Unknown)
		{
			int start_height = 50;

			// render progress
			float progress = target_user->mIDProgress / 100.;
			int max_width = 108;
			int bar_width = int(max_width*progress);
			if(bar_width == 0)
			{
				bar_width = 2;
			}

			int bar_height = 10;
			cv::Scalar bar_color(0, 0, 255);

			try {
				cv::Rect bg_patch = cv::Rect(bb.x, bb.y + start_height, 108, bar_height);
				cv::Mat roi = img(bg_patch);
				cv::Mat color(bg_patch.size(), CV_8UC3, cv::Scalar(0,0,0));
				double alpha = 0.5;
				cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);

				cv::Rect bar_patch = cv::Rect(bb.x, bb.y + start_height, bar_width, bar_height);
				img(bar_patch) = cv::Scalar(0,0,255);

			}
			catch (...) {
				// ...
			}

		}

		//user::TrackingConsistency tracking_status;
		//target_user->GetStatus(tracking_status);
		//if (tracking_status == user::TrackingConsistency_Uncertain) {
		//	bg_color = cv::Scalar(0, 14, 88);
		//}
		
		// ============================================= //
		// 5. DEBUG INFORMATION
		// ============================================= //
		
		if(mRenderDebug){
			// time for first prediction
			if (target_user->mTimeForFirstPrediction != 0) {
				std::string text_pred_speed = "First pred. in: " + std::to_string(target_user->mTimeForFirstPrediction) + "ms";
				cv::putText(img, text_pred_speed, cv::Point(bb.x + 108 + 10, bb.y - 108 + 10), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 0, 0), 1, 8);
			}

			// time since tracking init
			if (target_user->mTimeForFirstID != 0) {
				std::string text_id_speed = "Since init: " + std::to_string(target_user->mTimeForFirstID) + "ms";
				cv::putText(img, text_id_speed, cv::Point(bb.x + 108 + 10, bb.y - 108 + 30), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 0, 0), 1, 8);
			}

			// tracking consistency
			user::TrackingConsistency tracking_status;
			target_user->GetStatus(tracking_status);
			std::string text_tracking = "Tracking: ";
			cv::Scalar tracking_clr(0, 0, 0);
			if (tracking_status == user::TrackingConsistency_Uncertain) {
				text_tracking += "Unsafe";
				tracking_clr = cv::Scalar(0, 0, 255);
			}
			else {
				text_tracking += "Safe";
			}
			cv::putText(img, text_tracking, cv::Point(bb.x + 108 + 10, bb.y - 108 + 50), cv::FONT_HERSHEY_SIMPLEX, 0.38, tracking_clr, 1, 8);

			// pose
			if (target_user->GetFaceData(f))
			{
				int roll, pitch, yaw;
				f.GetEulerAngles(roll, pitch, yaw);
				std::string pose_text = "R: " + std::to_string(roll) + " | P: " + std::to_string(pitch) + " | Y: " + std::to_string(yaw);
				cv::putText(img, pose_text, cv::Point(bb.x + 108 + 10, bb.y - 108 + 70), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 0, 0), 1, 8);
			}

			// blur status

			// human status
			
		}

	}
}
