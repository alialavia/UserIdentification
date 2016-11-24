#include <opencv2/core.hpp>
#include <user/User.h>

using namespace user;

User::~User()
{

}
void User::SetUserID(int id, std::string nice_name)
{
	mUserID = id;
	mUserNiceName = nice_name;
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
void User::GetUserID(int& id, std::string& nice_name) const
{
	id = mUserID;
	nice_name = mUserNiceName;
}
cv::Rect2f User::GetFaceBoundingBox()
{
	return mFaceBoundingBox;
}
