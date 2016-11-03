#ifndef TRACKING__skeletontracker
#define TRACKING__skeletontracker

#include <base/UserIdentification.h>
#include <vector>
#include <opencv2/core/mat.hpp>

// Kinect SDK 2
#include <Kinect.h>

namespace tracking
{
	class SkeletonTracker
	{
	public:
		SkeletonTracker(IKinectSensor* sensor);
		~SkeletonTracker();
		HRESULT Init();
		HRESULT ExtractJoints(IBody* ppBodies[NR_USERS]);

		// data access
		int GetActiveBoundingBoxes(std::vector<cv::Rect2d>& boxes, std::vector<int>& user_ids) const;
		int GetJointsColorSpace(std::vector<std::vector<cv::Point2f>>& joints_colorspace, const DWORD joints, int srcWidth, int srcHeight, int outputWidth, int outputHeight) const;
		int GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, int srcWidth, int srcHeight, int outputWidth, int outputHeight, float box_size = 0.4) const;

	private:
		void reset();

		IKinectSensor* m_pKinectSensor;
		ICoordinateMapper* m_pCoordinateMapper;
		// user data
		std::vector<cv::Rect2d> mBoundingBoxes;
		// tracked user ids
		std::vector<int> mUserIDs;
		// raw joint data of the users
		Joint mUserJoints[NR_USERS][JointType_Count]; // joints in body space
	};

	class UserTracker
	{
		UserTracker(IKinectSensor* sensor);
		~UserTracker();
		HRESULT Init();
	private:
		IKinectSensor* pKinectSensor;
		SkeletonTracker* pSkeletonTracker;
	};
} // namespace

#endif
