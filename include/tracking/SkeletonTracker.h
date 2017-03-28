#ifndef TRACKING_SKELETONTRACKER_H_
#define TRACKING_SKELETONTRACKER_H_

#include <base/UserIdentification.h>
#include <vector>
#include <opencv2\opencv.hpp>

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

		void TrackVelocity(bool active);
		
		/// <summary>
		/// Extract joints from a IBody container (received from KinectInterface)
		/// </summary>
		/// <param name="ppBodies">Thebodies.</param>
		/// <returns>HRESULT.</returns>
		HRESULT ExtractJoints(IBody* ppBodies[NR_USERS], INT64 timestamp = 0);

		// --------------- data access

		size_t GetJointPosition(DWORD joint_type, std::vector<cv::Point3f>& joints);
		/// <summary>
		/// Get joint coordinates in image frame
		/// </summary>
		/// <param name="joint_coords">The joints colorspace.</param>
		/// <param name="joints">Select joints. E.g.: DWORD selection = base::JointType_Head | base::JointType_Neck;</param>
		/// <param name="space">Image space. E.g. base::ImageSpace_Color</param>
		/// <param name="outputWidth">Width of the output space.</param>
		/// <param name="outputHeight">Height of the output space.</param>
		/// <returns>int.</returns>
		size_t GetJointProjections(std::vector<std::vector<cv::Point2f>>& joint_coords, DWORD joints, base::ImageSpace space,
							   int outputWidth, int outputHeight) const;

		/// <summary>
		/// Gets the face bounding boxes.
		/// </summary>
		/// <param name="bounding_boxes">The bounding boxes.</param>
		/// <param name="space">Image space of the bounding boxes. E.g. base::ImageSpace_Color</param>
		/// <param name="outputWidth">Width of the output space.</param>
		/// <param name="outputHeight">Height of the output space.</param>
		/// <param name="box_size">Size of the box in meter.</param>
		/// <returns>Number of faces.</returns>
		size_t GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, std::vector<int> &user_ids, base::ImageSpace space, float box_size = 0.35) const;

		size_t GetFaceBoundingBoxesRobust(std::vector<cv::Rect2f>& bounding_boxes, std::vector<int> &user_ids, base::ImageSpace space, float box_size = 0.35) const;

		size_t GetUserSceneIDs(std::vector<int> &ids) const;
		// --------------- drawing methods
		
		/// <summary>
		/// Renders face bounding boxes onto the target image.
		/// </summary>
		/// <param name="target">The target image.</param>
		/// <param name="space">The image space.</param>
		/// <returns>HRESULT.</returns>
		HRESULT RenderFaceBoundingBoxes(cv::Mat &target, base::ImageSpace space) const;

		HRESULT RenderAllBodyJoints(cv::Mat &target, base::ImageSpace space) const;

		void ExtractFacesPatches(cv::Mat img, int patch_size, std::vector<cv::Mat> &patches, std::vector<int> &user_ids) const;

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

		bool mTrackVelocity;
		INT64 mCurrTime;
		INT64 mPrevTime;
		std::vector<int> mUserIDsBuffered;
		Joint mUserJointsBuffered[NR_USERS][JointType_Count]; // joints in body space - from last frame
		cv::Vec3d mJointVelocities[NR_USERS][JointType_Count];
	};


} // namespace

#endif
