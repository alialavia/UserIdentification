#ifndef USER_USER_H_
#define USER_USER_H_
#include <string>
#include <tracking\FaceTracker.h>
#include <opencv2/core.hpp>
#include <unordered_set>

namespace features
{
	class DlibFaceAligner;
}

namespace user
{

	enum IdentificationStatus
	{
		IDStatus_NonTarget = -1, // not considered yet (e.g. face has to be seen first)
		IDStatus_Unknown = 0,	 // has no ID yet
		IDStatus_Identified = 1, // has ID and is safe
		IDStatus_Uncertain = 2,	 // has ID but is not safe
		IDStatus_IsObject = 3	 // Object is tracked
	};

	enum ActionStatus
	{
		ActionStatus_Idle = 0,			// ready for new requests
		ActionStatus_Waiting = 1,		// do nothing, wait
		ActionStatus_WaitForCertainTracking = 2,
		ActionStatus_DataCollection = 3 //  for update/identification
	};

	// whether or not tracking instance is consistant/safe
	enum TrackingConsistency
	{
		TrackingConsistency_Uncertain = 0,	// tracking state alone unsafe
		TrackingConsistency_OK = 1
	};

	class User {

	public:
		User(
#ifdef _DLIB_PREALIGN
			features::DlibFaceAligner* aligner, 
#endif
			int tracking_id = -1,
			IdentificationStatus id_Status = IDStatus_Unknown
		) : mUserID(-1), mTrackingID(tracking_id), mUserNiceName(""), 
		// init user status
		mIDStatus(id_Status), mActionStatus(ActionStatus_Idle),
		mTrackingStatus(TrackingConsistency_OK), 
		mFaceData(nullptr), mUpdatingProfilePicture(false), mNrFramesNoFace(0), mNrFramesNoMovement(0)
#ifdef _DLIB_PREALIGN
			,pFaceAligner(aligner)
#endif
		{
#ifdef FACEGRID_RECORDING
			pGrid = new tracking::RadialFaceGrid(4, 6, 7);
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


		/////////////////////////////////////////////////
		/// 	Sampling

		bool TryToRecordFaceSample(const cv::Mat &scene_rgb);
		
		/////////////////////////////////////////////////
		/// 	Status

		void SetStatus(ActionStatus status);
		void SetStatus(IdentificationStatus status);
		void SetStatus(TrackingConsistency status);
		void GetStatus(IdentificationStatus &s1, ActionStatus &s2);
		void GetStatus(ActionStatus &s);
		void GetStatus(IdentificationStatus &s);
		void GetStatus(TrackingConsistency &s);

		/////////////////////////////////////////////////
		/// 	Identification

		void ResetUserIdentity(); // reset user completely/delete all user information
		void SetTrackingID(int id);
		void SetUserID(int id, std::string nice_name);
		void GetUserID(int& id, std::string& nice_name) const;
		int GetUserID() const;
		int GetTrackingID() const;
		void UpdateFaceBoundingBox(cv::Rect2f bb); // bounding box/position
		cv::Rect2f GetFaceBoundingBox();
		void SetPosition3D(const cv::Point3f &pos);
		cv::Point3f GetPosition3D() const;

		/////////////////////////////////////////////////
		/// 	Features

		void ResetSceneFeatures(); // reset all stored features (e.g. mFaceData). Performed after each frame
		void SetFaceData(tracking::Face f);
		bool GetFaceData(tracking::Face &f);

		/////////////////////////////////////////////////
		/// 	Helpers

		void PrintStatus();

		/////////////////////////////////////////////////
		/// 	Profile Picture

		bool IsViewedFromFront();
		bool NeedsProfilePicture(){return (mUpdatingProfilePicture ? false : mProfilePicture.empty());}
		void SetProfilePicture(cv::Mat picture){mProfilePicture = picture;}
		bool LooksPhotogenic();
		void SetPendingProfilePicture(bool status){mUpdatingProfilePicture = status;}
		bool GetProfilePicture(cv::Mat &pic);

		/////////////////////////////////////////////////
		/// 	Tracking Status

		void IncrementFaceDetectionStatus();
		std::string GetHumanStatusString();
		void IncrementBBMovementStatus();
		bool IsTrackingObject();
		bool IsHuman();

		// ids of possible confused users (for closed set identification)
		std::unordered_set<int> mClosedSetConfusionIDs;


		// temporal user prediction
		int mUserIDPredicted = 0;
		int mPredictionConfidence = 0;


	private:
		// user id and metadata
		int mUserID;
		std::string mUserNiceName;
		int mTrackingID;
		int mConfidence;

		// localization/tracking: must be set at all times
		cv::Rect2f mFaceBoundingBox;
		cv::Point2d mFaceCenter;
		cv::Point3f mPosition3D;	// position of spine center

		// status
		IdentificationStatus mIDStatus;
		ActionStatus mActionStatus;
		TrackingConsistency mTrackingStatus;

		// features: might be present or not
		tracking::Face* mFaceData;

		// profile picture
		cv::Mat mProfilePicture;
		bool mUpdatingProfilePicture;

		// nr of frames no face has been detected
		math::CircularBuffer<int> mBBMovement;
		int mIsObjectCombinedThresh = 90;
		int mIsObjectFaceThresh = 300;
		int mNrFramesNoFace;
		int mNrFramesNoMovement;

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