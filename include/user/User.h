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
		ActionStatus_UpdatePending = 3
	};

	class User {

	public:
		User() : mUserID(-1), mUserNiceName(""), mIDStatus(IDStatus_Unknown), mActionStatus(ActionStatus_Idle)
		{
#ifdef FACEGRID_RECORDING
			pGrid = new tracking::RadialFaceGrid(2, 10, 10);
#endif
		}
		~User()
		{
#ifdef FACEGRID_RECORDING
			delete(pGrid);
#endif
		}
		void SetUserID(int id, std::string nice_name);
		void SetIDStatus(IdentificationStatus status);
		void SetActionStatus(ActionStatus status);
		void SetFaceBoundingBox(cv::Rect2f bb);
		void SetFaceData(tracking::Face f);
		void GetStatus(IdentificationStatus &s1, ActionStatus &s2);
		void GetUserID(int& id, std::string& nice_name) const;
		cv::Rect2f GetFaceBoundingBox();
		tracking::Face GetFaceData();

		void PrintStatus()
		{
			std::cout << "--- id_status: " << mIDStatus << " | action: " << mActionStatus << std::endl;
		}

#ifdef FACEGRID_RECORDING
		tracking::RadialFaceGrid* pGrid;
#else
		// store image vector directly
#endif

	private:
		int mUserID;
		std::string mUserNiceName;

		// status
		IdentificationStatus mIDStatus;
		ActionStatus mActionStatus;

		// current user data
		cv::Rect2f mFaceBoundingBox;
		tracking::Face mFaceData;
	};


}

#endif