#ifndef TRACKING_FACETRACKER_H_
#define TRACKING_FACETRACKER_H_

#include <base/UserIdentification.h>
#include <Kinect.Face.h>
#include <vector>
#include <opencv2\opencv.hpp>
#include <io/KinectInterface.h>
#include "math/Math.h"

#include "io/ImageHandler.h"

#define _DEBUG_FACETRACKER

namespace tracking
{

	class RadialFaceGrid  {
	public:

		RadialFaceGrid(
			size_t interv_r_ = 3,
			size_t interv_p_ = 6,
			size_t interv_y_ = 10
		): 
			 interv_r(interv_r_),
			 interv_p(interv_p_),
			 interv_y(interv_y_),
			 image_grid(interv_r_, interv_p_, interv_y_)	// init storage container
		{
			// calculate index mapping functions
			a_r = (float)(interv_r - 1) / (cRMax - cRMin);
			a_p = (float)(interv_p - 1) / (cPMax - cPMin);
			a_y = (float)(interv_y - 1) / (cYMax - cYMin);
			b_r = -a_r*cRMin;
			b_p = -a_p*cPMin;
			b_y = -a_y*cYMin;
		}

		~RadialFaceGrid() {

		}

		void DumpImageGrid(std::string filename = "capture", std::string log_name = "face_log.csv", std::string out_folder = "face_grid");
		std::vector<cv::Mat*> ExtractGrid();
		void GetFaceGridPitchYaw(cv::Mat &dst, int canvas_height=500);

		bool IsFree(int roll, int pitch, int yaw) {
			// check if we already got an image at this position
			int iroll = iRoll(roll);
			int	ipitch = iPitch(pitch);
			int iyaw = iYaw(yaw);
			return image_grid.IsFree(iroll, ipitch, iyaw);
		}

		// throws exception if pose out of range
		bool StoreSnapshot(int roll, int pitch, int yaw, const cv::Mat &face)
		{
			int iroll = iRoll(roll);
			int	ipitch = iPitch(pitch);
			int iyaw = iYaw(yaw);

			// save image
			cv::Mat * ptr = image_grid.CopyTo(iroll, ipitch, iyaw, face);
			cv::Vec3d ang = cv::Vec3d(roll, pitch, yaw);

			// store rotation
			angles[ptr] = ang;
			return true;
		}

		void ResizeImages(int size);

		void Clear()
		{
			image_grid.Reset();
			angles.clear();
		}

		// ---------- index mapper
		int iRoll(int roll) {
			return floor(a_r*roll +b_r);
		}
		int iPitch(int pitch) {
			return floor(a_p*pitch + b_p);
		}
		int iYaw(int yaw) {
			return floor(a_y*yaw + b_y);
		}

		// images
		math::Array3D<cv::Mat> image_grid;

		// array3d index to precies angles
		std::map<cv::Mat*, cv::Vec3d> angles;

		size_t interv_r;
		size_t interv_p;
		size_t interv_y;

		// image grid resolution
		const int cRMin = -70;
		const int cRMax = 70;
		const int cPMin = -30;
		const int cPMax = 40;
		const int cYMin = -50;
		const int cYMax = 50;

		// index mapper
		float a_r;
		float b_r;
		float a_p;
		float b_p;
		float a_y;
		float b_y;
	};


	class Face
	{

	public:
		Face() {

		}
		cv::Rect2f boundingBox;
		cv::Rect2f boundingBoxIR;
		cv::Vec4f Rotation;	// rotation quaternion x,y,z,w
		PointF Points[FacePointType::FacePointType_Count];
		PointF PointsIR[FacePointType::FacePointType_Count];
		DetectionResult Properties[FaceProperty::FaceProperty_Count];

		void GetEulerAngles(int& roll, int& pitch, int& yaw) {
			double x = Rotation[0];
			double y = Rotation[1];
			double z = Rotation[2];
			double w = Rotation[3];
			// convert face rotation quaternion to Euler angles in degrees		
			double dPitch, dYaw, dRoll;
			dPitch = atan2(2 * (y * z + w * x), w * w - x * x - y * y + z * z) / M_PI * 180.0;
			dYaw = asin(2 * (w * y - x * z)) / M_PI * 180.0;
			dRoll = atan2(2 * (x * y + w * z), w * w + x * x - y * y - z * z) / M_PI * 180.0;
			const double c_FaceRotationIncrementInDegrees = 5.0f;
			// clamp rotation values in degrees to a specified range of values to control the refresh rate
			double increment = c_FaceRotationIncrementInDegrees;
			pitch = static_cast<int>(floor((dPitch + increment / 2.0 * (dPitch > 0 ? 1.0 : -1.0)) / increment) * increment);
			yaw = static_cast<int>(floor((dYaw + increment / 2.0 * (dYaw > 0 ? 1.0 : -1.0)) / increment) * increment);
			roll = static_cast<int>(floor((dRoll + increment / 2.0 * (dRoll > 0 ? 1.0 : -1.0)) / increment) * increment);
		}
	};


	class FaceTracker
	{
	public:
		FaceTracker(IKinectSensor* sensor):
		m_pKinectSensor(sensor)
		{
			
		}

		HRESULT ExtractFacialData(FaceData face_data[NR_USERS]);

		int GetFaceBoundingBoxesRobust(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space) const;
		int GetUserSceneIDs(std::vector<int> &ids) const;
		int GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space) const;
		std::vector<Face> GetFaces();
		
		HRESULT RenderFaceBoundingBoxes(cv::Mat &target, base::ImageSpace space) const;
		HRESULT RenderFaceFeatures(cv::Mat &target, base::ImageSpace space) const;

	private:
		IKinectSensor* m_pKinectSensor;
		// tracked user ids
		std::vector<int> mUserIDs;
		std::vector<Face> mFaces;

	};

}

#endif