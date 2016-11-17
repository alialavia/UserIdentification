#include <user\UserManager.h>
#include <user\User.h>
#include <opencv2/imgproc.hpp>


using namespace  user;

bool UserManager::Init(io::TCPClient* connection)
{
	if (connection == nullptr)
	{
		return false;
	}
	pServerConn = connection;
	return true;
}


// refresh tracked users: scene_id, bounding boxes
void UserManager::RefreshTrackedUsers(const std::vector<int> &user_scene_ids)
{
	// update existing users - remove non tracked
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		if (std::find(user_scene_ids.begin(), user_scene_ids.end(), it->first) != user_scene_ids.end())
		{
			// user is in scene - update positional data
		}
		else
		{
			// user has left scene - delete tracking instance
			delete(it->second);
			// remove mapping
			mFrameIDToUser.erase(it);
		}
	}

	// add new users
	for (int i = 0; i<user_scene_ids.size(); i++)
	{
		int scene_id = user_scene_ids[i];
		// user not tracked yet - initiate new user model
		if (mFrameIDToUser.count(scene_id) == 0)
		{
			mFrameIDToUser[scene_id] = new User();
		}
	}
}

// incorporate processed requests: update user ids
void UserManager::ApplyUserIdentification()
{
	// handle processed requests


	// apply to users
}

// send identification requests for all unknown users
void UserManager::RequestUserIdentification()
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		if (it->second->GetIDStatus() == IDStatus_Unknown)
		{
			// TODO: make identification request

			// send request id
			//pServerConn->SendChar();


			// send payload
			it->second->SetIDStatus(IDStatus_Pending);
		}
	}
}

// ----------------- helper functions

void UserManager::DrawUsers(cv::Mat &img)
{
	for (auto it = mFrameIDToUser.begin(); it != mFrameIDToUser.end(); ++it)
	{
		cv::Rect bb = it->second->GetFaceBoundingBox();

		// draw face bounding box
		cv::rectangle(img, bb, cv::Scalar(0, 0, 255), 2, cv::LINE_4);


		// draw identification status
		float font_size = 0.5;
		std::string text;
		enum IdentificationStatus status = it->second->GetIDStatus();
		if (status == IDStatus_Identified)
		{
			text = "Status: identified";
		}
		else if (status == IDStatus_Pending)
		{
			text = "Status: pending";
		}
		else
		{
			text = "Status: unknown";
		}
		cv::putText(img, text, cv::Point(bb.x, bb.y), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0, 0, 0), 1, 8);
	}
}
