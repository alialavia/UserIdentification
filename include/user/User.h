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

	// whether or not we are tracking a person
	enum HumanTrackingStatus
	{
		HumanTrackingStatus_Uncertain = 0,
		HumanTrackingStatus_Certain = 1
	};

	class User {

	public:
		User(
#ifdef _DLIB_PREALIGN
			features::DlibFaceAligner* aligner
#endif
		) : mUserID(-1), mUserNiceName(""), 
		// init user status
		mIDStatus(IDStatus_Unknown), mActionStatus(ActionStatus_Idle), 
		mTrackingStatus(TrackingConsistency_OK), 
		mFaceData(nullptr), mUpdatingProfilePicture(false), mConfidence(0), mNrFramesNoFace(0), mNrFramesNoMovement(0)
#ifdef _DLIB_PREALIGN
			,pFaceAligner(aligner)
#endif
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
		void ResetUserIdentity();

		// ========= user status
		// setters
		void SetStatus(ActionStatus status);
		void SetStatus(IdentificationStatus status);
		void SetStatus(TrackingConsistency status);
		void SetStatus(HumanTrackingStatus status);
		// getters
		void GetStatus(IdentificationStatus &s1, ActionStatus &s2);
		void GetStatus(ActionStatus &s);
		void GetStatus(IdentificationStatus &s);
		void GetStatus(TrackingConsistency &s);
		void GetStatus(HumanTrackingStatus &s);

		// ========= identification
		// id
		void SetUserID(int id, std::string nice_name);
		void GetUserID(int& id, std::string& nice_name) const;
		int GetUserID() const;
		// bounding box/position
		void UpdateFaceBoundingBox(cv::Rect2f bb);
		cv::Rect2f GetFaceBoundingBox();

		// take snapshot of face
		bool TryToRecordFaceSample(const cv::Mat &scene_rgb);

		// ========= features
		// reset all stored features (e.g. mFaceData). Performed after each frame
		void ResetSceneFeatures();
		void SetFaceData(tracking::Face f);
		bool GetFaceData(tracking::Face &f);

		// ========= helpers
		void PrintStatus();

		// ========= profile picture utilities
		bool IsViewedFromFront();
		bool NeedsProfilePicture(){return (mUpdatingProfilePicture ? false : mProfilePicture.empty());}
		void SetProfilePicture(cv::Mat picture){mProfilePicture = picture;}
		bool LooksPhotogenic()
		{
			tracking::Face face;
			bool succ = false;
			if (GetFaceData(face)) {
				if (face.IsPhotogenic())
				{
					succ = true;
				}
			}
			return succ;
		}
		void SetPendingProfilePicture(bool status){mUpdatingProfilePicture = status;}
		bool GetProfilePicture(cv::Mat &pic);

		// identification confidence
		int GetConfidence() { return mConfidence; }
		void SetConfidence(const int &conf) { mConfidence = conf; }

		// ========= Human tracking status
		void IncrementFaceDetectionStatus() {
			if (mFaceData != nullptr) {
				mNrFramesNoFace = 0;
			}
			else {
				mNrFramesNoFace++;
			}
		}

		std::string GetHumanStatusString() {
			return " Face: "+std::to_string(mNrFramesNoFace)+" | Movement: "+std::to_string(mNrFramesNoMovement);
		}

		void IncrementBBMovementStatus() {
			int thresh = 0;
			float median = 1000.;
			if (mBBMovement.FullMedian(median)) {
				if (median <= thresh) {
					mNrFramesNoMovement++;
				}
				else {
					mNrFramesNoMovement = 0;
				}
			}
			else {
			}
		}

		bool IsTrackingObject() {
			if (
				(
				mNrFramesNoFace > mIsObjectCombinedThresh && mNrFramesNoMovement > mIsObjectCombinedThresh) ||
				mNrFramesNoFace > mIsObjectFaceThresh
				) {
				return true;
			}
			return false;
		}


	private:
		// user id
		int mUserID;
		std::string mUserNiceName;
		// localization/tracking: must be set at all times
		cv::Rect2f mFaceBoundingBox;
		cv::Point2d mFaceCenter;

		// status
		IdentificationStatus mIDStatus;
		ActionStatus mActionStatus;
		TrackingConsistency mTrackingStatus;

		// features: might be present or not
		tracking::Face* mFaceData;

		// profile picture
		cv::Mat mProfilePicture;
		bool mUpdatingProfilePicture;

		// current confidence of the identification in %
		int mConfidence;

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