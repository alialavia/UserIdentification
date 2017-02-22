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


}

void StreamUserManager::GenerateRequests(cv::Mat scene_rgb)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		IdentificationStatus id_status;
		ActionStatus action;
		user::User* target_user = it->second;
		target_user->GetStatus(id_status, action);
	}

}