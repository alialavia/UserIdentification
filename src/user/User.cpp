#include <user/User.h>

using namespace user;


void User::SetUserID(int id, std::string nice_name)
{
	mUserID = id;
	mUserNiceName = nice_name;
	mIDStatus = IDStatus_Identified;
}
void User::SetIDStatus(IdentificationStatus status)
{
	mIDStatus = status;
}
void User::SetActionStatus(ActionStatus status) {
	mActionStatus = status;
}
void User::SetFaceBoundingBox(cv::Rect2f bb)
{
	mFaceBoundingBox = bb;
}
void User::SetFaceData(tracking::Face f)
{
	// allocate new face
	mFaceData = new tracking::Face(f);
}
void User::GetStatus(IdentificationStatus &s1, ActionStatus &s2)
{
	s1 = mIDStatus;
	s2 = mActionStatus;
}

cv::Rect2f User::GetFaceBoundingBox()
{
	return mFaceBoundingBox;
}
bool User::GetFaceData(tracking::Face& f)
{
	if (mFaceData == nullptr) {
		return false;
	}
	f = *mFaceData;
	return true;
}
