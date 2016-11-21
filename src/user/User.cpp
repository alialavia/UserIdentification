#include <opencv2/core.hpp>
#include <user/User.h>

using namespace user;

User::~User()
{

}
void User::SetUserID(int id)
{
	mUserID = id;
	mIDStatus = IDStatus_Identified;
}
void User::SetIDStatus(enum IdentificationStatus status)
{
	mIDStatus = status;
}
void User::SetFaceBoundingBox(cv::Rect2f bb)
{
	mFaceBoundingBox = bb;
}
enum IdentificationStatus User::GetIDStatus()
{
	return mIDStatus;
}
int User::GetUserID()
{
	return mUserID;
}
cv::Rect2f User::GetFaceBoundingBox()
{
	return mFaceBoundingBox;
}
