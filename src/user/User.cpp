#include <user/User.h>
#include "features/Face.h"


using namespace user;

/////////////////////////////////////////////////
/// 	Sampling


bool User::TryToRecordFaceSample(const cv::Mat &scene_rgb)
{
	tracking::Face face;
	bool succ = false;
	if (GetFaceData(face)) {
		int roll, pitch, yaw;
		face.GetEulerAngles(roll, pitch, yaw);
		if (pGrid->IsFree(roll, pitch, yaw))
		{
			cv::Rect2f facebb = GetFaceBoundingBox();
			cv::Mat face_snap = scene_rgb(facebb);
			cv::Mat aligned;
			// TODO: use Microsoft API with Face data
#ifdef _DLIB_PREALIGN
			if (pFaceAligner->AlignImage(96, face_snap, aligned))
#else
			aligned = face_snap;
			// resize (requests needs all squared with same size)
			cv::resize(aligned, aligned, cv::Size(120, 120));
#endif
			{
				try
				{
					pGrid->StoreSnapshot(roll, pitch, yaw, aligned);
					succ = true;
				}
				catch (...)
				{
				}
			}
		}

	}
	return succ;
}


/////////////////////////////////////////////////
/// 	Status

void User::SetStatus(ActionStatus status) {
	mActionStatus = status;
}
void User::SetStatus(IdentificationStatus status)
{
	mIDStatus = status;
	if(status == IDStatus_Identified)
	{
		mClosedSetConfusionIDs.clear();
	}
}
void User::SetStatus(RequestStatus status) {
	mRequestStatus = status;
}
void User::SetStatus(TrackingConsistency status)
{
	mTrackingStatus = status;
}
void User::GetStatus(IdentificationStatus &s1, ActionStatus &s2)
{
	s1 = mIDStatus;
	s2 = mActionStatus;
}
void User::GetStatus(ActionStatus &s)
{
	s = mActionStatus;
}
void User::GetStatus(IdentificationStatus &s)
{
	s = mIDStatus;
}
void User::GetStatus(TrackingConsistency &s)
{
	s = mTrackingStatus;
}
void User::GetStatus(RequestStatus &s)
{
	s = mRequestStatus;
}

/////////////////////////////////////////////////
/// 	Identification

void User::ResetUserIdentity() {
	mUserID = -1;
	mIDStatus = IDStatus_Unknown;
	mActionStatus = ActionStatus_Idle;
	mTrackingStatus = TrackingConsistency_OK;
	mUpdatingProfilePicture = false;

	mUserIDPredicted = 0;
	mPredictionConfidence = 0;
	mIDProgress = 0;

	// release profile image
	if (!mProfilePicture.empty())
	{
		mProfilePicture.release();
	}

	mClosedSetConfusionIDs.clear();

#ifdef FACEGRID_RECORDING
	// reset face grid
	pGrid->Clear();
#endif

}

void User::SetTrackingID(int id) {
	mTrackingID = id;
}

int User::GetTrackingID() const{
	return mTrackingID;
}

void User::SetUserID(int id, std::string nice_name)
{
	mUserID = id;
	mUserNiceName = nice_name;
	mIDStatus = IDStatus_Identified;
	mIDProgress = 100;
	mClosedSetConfusionIDs.clear();
}

void User::GetUserID(int& id, std::string& nice_name) const
{
	id = mUserID;
	nice_name = mUserNiceName;
}


int User::GetUserID() const
{
	return mUserID;
}

void User::UpdateFaceBoundingBox(cv::Rect2f bb)
{
	// bb centroid
	cv::Point2d centroid = cv::Point2d((bb.x + bb.width) / 2, (bb.y + bb.height) / 2);

	if (mFaceBoundingBox.width > 0) {
		// add bb movement
		mBBMovement.AddElement(cv::norm(mFaceCenter - centroid));
		mFaceCenter = centroid;
		mFaceBoundingBox = bb;
	}
	else {
		// first round
		mFaceCenter = centroid;
		mFaceBoundingBox = bb;
	}
}

cv::Rect2f User::GetFaceBoundingBox()
{
	return mFaceBoundingBox;
}

void User::SetPosition3D(const cv::Point3f &pos) {
	mPosition3D = pos;
}

cv::Point3f User::GetPosition3D() const {
	return mPosition3D;
}


/////////////////////////////////////////////////
/// 	Features

void User::ResetSceneFeatures() {
	// reset all stored features
	if (mFaceData != nullptr) {
		delete(mFaceData);
		mFaceData = nullptr;
	}
}

void User::SetFaceData(tracking::Face f)
{
	// allocate new face
	mFaceData = new tracking::Face(f);
}

bool User::GetFaceData(tracking::Face& f)
{
	if (mFaceData == nullptr) {
		return false;
	}
	f = *mFaceData;
	return true;
}

/////////////////////////////////////////////////
/// 	Helpers

void User::PrintStatus()
{
	std::cout << "--- id_status: " << mIDStatus << " | action: " << mActionStatus << std::endl;
}


/////////////////////////////////////////////////
/// 	Profile Picture

/*
*
// Client Side Picture Taking
- During update (when person has been identified)
- If person has no profile picture yet (nothing received from server)
- Evaluate the face (bb) in each frame if the orientation is frontal
- if it is approx. frontal: optionally rotate
- save picture to user instance
- send profilePictureUpdate Request to server

// difficulties
- profile picture taken, when tracking switches

// solutions
- also classify profile picture and reject if it does not comply with corresponding ID
*/

bool User::IsViewedFromFront()
{
	// get face orientation
	if (mFaceData != nullptr) {
		// calc euler angles
		int roll, pitch, yaw;
		mFaceData->GetEulerAngles(roll, pitch, yaw);
		// optional: rotate image

		if (pitch >= 30 || pitch <= -30)
		{
			return false;
		}
		if (roll >= 30 || roll <= -30)
		{
			return false;
		}
		if (yaw >= 30 || yaw <= -30)
		{
			return false;
		}

		return true;
	}
	return false;
}

bool User::GetProfilePicture(cv::Mat &pic)
{
	if (mProfilePicture.empty())
	{
		return false;
	}
	pic = mProfilePicture;
	return true;
}

bool User::LooksPhotogenic()
{
	tracking::Face face;
	bool succ = false;
	if (GetFaceData(face)) {
		if (face.IsPhotogenic())
		{
			succ = true;
		}
	}
	return succ;
}


/////////////////////////////////////////////////
/// 	Tracking Status

void User::IncrementFaceDetectionStatus() {
	if (mFaceData != nullptr) {
		mNrFramesNoFace = 0;
	}
	else {
		mNrFramesNoFace++;
	}
}

std::string User::GetHumanStatusString() {
	return " Face: " + std::to_string(mNrFramesNoFace) + " | Movement: " + std::to_string(mNrFramesNoMovement);
}

std::string User::GetActionStatusString()
{
	if(mActionStatus == ActionStatus_WaitForCertainTracking)
	{
		std::string text = "Re-ID | IDs: ";

		for (auto conf_id : mClosedSetConfusionIDs) {
			text +=std::to_string(conf_id) + " ";
		}

		return text;
	}else if(mActionStatus == ActionStatus_Waiting)
	{
		return "Waiting";
	}
	else if (mActionStatus == ActionStatus_DataCollection)
	{
		std::string pending = (mRequestStatus == RequestStatus_Pending? "P": "I");
		return "Sampling: (" + std::to_string(pGrid->nr_images()) + ") " + pending;
	}
	else
	{
		return "Idle";
	}
}

void User::IncrementBBMovementStatus() {
	int thresh = 0;
	float median = 1000.;
	if (mBBMovement.FullMedian(median)) {
		if (median <= thresh) {
			mNrFramesNoMovement++;
		}
		else {
			mNrFramesNoMovement = 0;
		}
	}
	else {
	}
}

bool User::IsTrackingObject() {
	if (
		(
			mNrFramesNoFace > mIsObjectCombinedThresh && mNrFramesNoMovement > mIsObjectCombinedThresh) ||
		mNrFramesNoFace > mIsObjectFaceThresh
		) {
		return true;
	}
	return false;
}

bool User::IsHuman() {
	if (mFaceData != nullptr) {
		return true;
	}
	else {
		return false;
	}
}