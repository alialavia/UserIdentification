#ifndef USER_USER_H_
#define USER_USER_H_
#include <string>
#include <tracking\FaceTracker.h>
#include <opencv2/core.hpp>

namespace features
{
	class DlibFaceAligner;
}

namespace user
{

	enum IdentificationStatus
	{
		IDStatus_Unknown = 0,	 // has no ID yet
		IDStatus_Identified = 1, // has ID and is safe
		IDStatus_Uncertain = 2	 // has ID but is not safe
	};

	enum ActionStatus
	{
		ActionStatus_Idle = 0,			// ready for new requests
		ActionStatus_Waiting = 1,		// do nothing, wait
		ActionStatus_WaitForCertainTracking = 2,
		ActionStatus_DataCollection = 3 //  for update/identification
	};
#ifdef _CHECK_TRACKING_CONF
	// whether or not tracking instance is consistant/safe
	enum TrackingStatus
	{
		TrackingStatus_Uncertain = 0,	// tracking state alone unsafe
		TrackingStatus_Certain = 1
	};
#endif
	// whether or not we are tracking a person
	enum HumanTrackingStatus
	{
		HumanTrackingStatus_Uncertain = 0,
		HumanTrackingStatus_Certain = 1
	};

	class User {

	public:
		User() : mUserID(-1), mUserNiceName(""), 
		// init user status
		mIDStatus(IDStatus_Unknown), mActionStatus(ActionStatus_Idle), 
#ifdef _CHECK_TRACKING_CONF
		mTrackingStatus(TrackingStatus_Certain), 
#endif
		mHumanTrackingStatus(HumanTrackingStatus_Certain),
			mFaceData(nullptr), mUpdatingProfilePicture(false), mConfidence(0)
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
#ifdef _CHECK_TRACKING_CONF
		void SetStatus(TrackingStatus status);
#endif
		void SetStatus(HumanTrackingStatus status);
		// getters
		void GetStatus(IdentificationStatus &s1, ActionStatus &s2);
		void GetStatus(ActionStatus &s);
		void GetStatus(IdentificationStatus &s);
#ifdef _CHECK_TRACKING_CONF
		void GetStatus(TrackingStatus &s);
#endif
		void GetStatus(HumanTrackingStatus &s);

		// ========= identification
		// id
		void SetUserID(int id, std::string nice_name);
		void GetUserID(int& id, std::string& nice_name) const;
		int GetUserID() const;
		// bounding box/position
		void SetFaceBoundingBox(cv::Rect2f bb);
		cv::Rect2f GetFaceBoundingBox();

		// take snapshot of face
		bool TryToRecordFaceSample(const cv::Mat &scene_rgb);

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
		bool mUpdatingProfilePicture;

		// current confidence of the identification in %
		int mConfidence;

		// dlib face aligner
#ifdef _DLIB_PREALIGN
		features::DlibFaceAligner* pFaceAligner;
#endif

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