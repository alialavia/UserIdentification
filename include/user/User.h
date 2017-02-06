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
			mFaceData(nullptr)
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