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
	// whether or not tracking instance is consistant/safe
	enum TrackingStatus
	{
		TrackingStatus_Uncertain = 0,
		TrackingStatus_Certain = 1
	};
	// whether or not we are tracking a person
	enum HumanTrackingStatus
	{
		HumanTrackingStatus_Uncertain = 0,
		HumanTrackingStatus_Certain = 1
	};

	class User {

	public:
		User() : mUserID(-1), mUserNiceName(""), mIDStatus(IDStatus_Unknown), mActionStatus(ActionStatus_Idle),
			mFaceData(nullptr), mUpdatingProfilePicture(false), mConfidence(0), mTrackingIsSafe(true)
		{
#ifdef FACEGRID_RECORDING
			pGrid = new tracking::RadialFaceGrid(2, 15, 15);
#endif
		}
		~User()
		{
			// data freed in destructor of grid
#ifdef FACEGRID_RECORDING
			delete(pGrid);
#endif
			// delete allocated features
			ResetSceneFeatures();
		}

		// ========= general
		// reset user completely/delete all user information
		void ResetUser();

		// ========= user status
		// setters
		void SetStatus(ActionStatus status);
		void SetStatus(IdentificationStatus status);
		void SetStatus(TrackingStatus status);
		void SetStatus(HumanTrackingStatus status);
		// getters
		void GetStatus(IdentificationStatus &s1, ActionStatus &s2);
		void GetStatus(ActionStatus &s);
		void GetStatus(IdentificationStatus &s);
		void GetStatus(TrackingStatus &s);
		void GetStatus(HumanTrackingStatus &s);

		// ========= identification
		// id
		void SetUserID(int id, std::string nice_name);
		void GetUserID(int& id, std::string& nice_name) const;
		int GetUserID() const;
		// bounding box/position
		void SetFaceBoundingBox(cv::Rect2f bb);
		cv::Rect2f GetFaceBoundingBox();

		// ========= features
		// reset all stored features (e.g. mFaceData)
		void ResetSceneFeatures();
		void SetFaceData(tracking::Face f);
		bool GetFaceData(tracking::Face &f);

		// ========= helpers
		void PrintStatus();

		// ========= profile picture utilities
		bool IsViewedFromFront();
		bool NeedsProfilePicture(){return (mUpdatingProfilePicture ? false : mProfilePicture.empty());}
		void SetProfilePicture(cv::Mat picture){mProfilePicture = picture;}
		void SetPendingProfilePicture(bool status){mUpdatingProfilePicture = status;}
		bool GetProfilePicture(cv::Mat &pic);

		// identification confidence
		int GetConfidence() { return mConfidence; }
		void SetConfidence(const int &conf) { mConfidence = conf; }

		// tracking status
		void SetTrackingIsSafe(bool is_save) {mTrackingIsSafe = is_save;}
		bool TrackingIsSafe() {return mTrackingIsSafe;}


	private:
		// user id
		int mUserID;
		std::string mUserNiceName;
		// localization/tracking: must be set at all times
		cv::Rect2f mFaceBoundingBox;

		// status
		IdentificationStatus mIDStatus;
		ActionStatus mActionStatus;
		TrackingStatus mTrackingStatus;
		HumanTrackingStatus mHumanTrackingStatus;

		// features: might be present or not
		tracking::Face* mFaceData;

		// profile picture
		cv::Mat mProfilePicture;

		// todo:delete
		bool mUpdatingProfilePicture;

		// current confidence of the identification in %
		int mConfidence;
		// if the tracking is confident (no bounding boxes overlap)
		// Todo: refacotr = Tracking Status
		bool mTrackingIsSafe;

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