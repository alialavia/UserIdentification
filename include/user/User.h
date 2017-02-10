#ifndef USER_USER_H_
#define USER_USER_H_
#include <string>
#include <tracking\FaceTracker.h>
#include <opencv2/core.hpp>

namespace user
{

	enum IdentificationStatus
	{
		IDStatus_Unknown = 0,
		IDStatus_Identified = 1
	};

	enum ActionStatus
	{
		ActionStatus_Idle = 0,
		ActionStatus_IDPending = 1,
		ActionStatus_Initialization = 2,
		ActionStatus_UpdatePending = 3,
		ActionStatus_DataCollection = 4
	};

	class User {

	public:
		User() : mUserID(-1), mUserNiceName(""), mIDStatus(IDStatus_Unknown), mActionStatus(ActionStatus_Idle),
			mFaceData(nullptr), mUpdatingProfilePicture(false), mConfidence(0)
		{
#ifdef FACEGRID_RECORDING
			pGrid = new tracking::RadialFaceGrid(2, 15, 15);
#endif
		}
		~User()
		{
#ifdef FACEGRID_RECORDING
			delete(pGrid);
#endif
			// delete allocated features
			ResetSceneFeatures();
		}

		// ------ setters
		// status/id setters
		void SetUserID(int id, std::string nice_name);
		void SetIDStatus(IdentificationStatus status);
		void SetActionStatus(ActionStatus status);
		void SetFaceBoundingBox(cv::Rect2f bb);
		// feature setters
		void SetFaceData(tracking::Face f);

		void ResetSceneFeatures() {
			// reset all stored features
			if (mFaceData != nullptr) {
				delete(mFaceData);
				mFaceData = nullptr;
			}
		}

		void ResetUser() {
			mUserID = -1;
			mIDStatus = IDStatus_Unknown;
			mActionStatus = ActionStatus_Idle;
			// release profile image
			if(!mProfilePicture.empty())
			{
				mProfilePicture.release();
			}
		}

		// ------ getters
		// status/id getters
		void GetStatus(IdentificationStatus &s1, ActionStatus &s2);
		void GetUserID(int& id, std::string& nice_name) const
		{
			id = mUserID;
			nice_name = mUserNiceName;
		}
		int GetUserID()
		{
			return mUserID;
		}

		cv::Rect2f GetFaceBoundingBox();
		// feature getters
		bool GetFaceData(tracking::Face &f);

		// ------ helpers

		void PrintStatus()
		{
			std::cout << "--- id_status: " << mIDStatus << " | action: " << mActionStatus << std::endl;
		}

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

		bool IsViewedFromFront()
		{
			// get face orientation
			if (mFaceData != nullptr) {
				// calc euler angles
				int roll, pitch, yaw;
				mFaceData->GetEulerAngles(roll, pitch, yaw);
				// optional: rotate image

				if(pitch >= 30 || pitch <= -30)
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

		bool NeedsProfilePicture()
		{
			return (mUpdatingProfilePicture ? false : mProfilePicture.empty());
		}
		void AssignProfilePicture(cv::Mat picture)
		{
			mProfilePicture = picture;
		}
		void SetPendingProfilePicture(bool status)
		{
			mUpdatingProfilePicture = status;
		}
		bool GetProfilePicture(cv::Mat &pic)
		{
			if(mProfilePicture.empty())
			{
				return false;
			}
			pic = mProfilePicture;
			return true;
		}

		int GetConfidence()
		{
			return mConfidence;
		}
		void SetConfidence(const int &conf)
		{
			mConfidence = conf;
		}

	private:
		// user id
		int mUserID;
		std::string mUserNiceName;
		// localization/tracking: must be set at all times
		cv::Rect2f mFaceBoundingBox;
		// status
		IdentificationStatus mIDStatus;
		ActionStatus mActionStatus;

		// features: might be present or not
		tracking::Face* mFaceData;

		cv::Mat mProfilePicture;
		bool mUpdatingProfilePicture;

		int mConfidence;

	public:
		// temporal model data (images, accumulated status)
#ifdef FACEGRID_RECORDING
		tracking::RadialFaceGrid* pGrid;
#else
		std::vector<cv::Mat> mImages;
#endif


	};


}

#endif