#ifndef TRACKING_FACETRACKER_H_
#define TRACKING_FACETRACKER_H_

#include <base/UserIdentification.h>
#include <Kinect.Face.h>
#include <vector>
#include <opencv2\opencv.hpp>
#include <io/KinectInterface.h>
#include "math/Math.h"


#define _DEBUG_FACETRACKER

namespace tracking
{


	class RadialFaceGrid  {
	public:

		RadialFaceGrid(): 
		image_grid(nr_steps, nr_steps, nr_steps),
		step_size_r((roll_max - roll_min)/nr_steps),
		step_size_p((pitch_max - pitch_min)/nr_steps),
		step_size_y((yaw_max - yaw_min)/nr_steps)
		{

		}

		~RadialFaceGrid() {

		}

		void AllocateGrid() {
			//}

		}

		void DumpImageGrid() {
			// save the image grid to the hard drive

		}



		bool IsOccupied(int roll, int pitch, int yaw) {
			// check if we already got an image at this position
			int iroll, ipitch, iyaw;
			bool occupied = GetRadiantIndices(roll, pitch, yaw, iroll, ipitch, iyaw);

			if(!occupied)
			{
				occupied = image_grid.IsFree(iroll, ipitch, iyaw);
			}

			return occupied;

		}

		bool StoreSnapshot(int roll, int pitch, int yaw, cv::Mat face)
		{
			// store snapshot in the image grid

		}

		bool GetRadiantIndices(int roll, int pitch, int yaw, int& iRoll, int& iYaw, int& iPitch) {
			// calculate radiant indices for angles

			bool r = false;

			// check range
			r = (roll >= roll_min && roll <= roll_max &&
				pitch >= pitch_min && pitch <= pitch_max &&
				yaw >= yaw_min && yaw <= yaw_max);

			// calc indices
			if (r) {
				iRoll = floor(roll/step_size_r);
				iYaw = floor(pitch/step_size_p);
				iPitch = floor(yaw/step_size_y);
			}

			return r;
		}


		// images
		math::Array3D<cv::Mat> image_grid;


		const int nr_steps = 10;

		const float step_size_r;
		const float step_size_p;
		const float step_size_y;


		const int roll_min = 0;
		const int roll_max = 180;
		const int pitch_min = 0;
		const int pitch_max = 180;
		const int yaw_min = 0;
		const int yaw_max = 180;
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

		HRESULT ExtractFacialData(FaceData face_data[NR_USERS])
		{

			HRESULT hr = E_FAIL;
			// reset tracking data
			mUserIDs.clear();
			mFaces.clear();

			for (int iFace = 0; iFace < NR_USERS; ++iFace)
			{
				FaceData fd = face_data[iFace];
				if (fd.tracked) {
					// new face container
					Face face_container;

					// bounding boxes
					face_container.boundingBox = cv::Rect2f(
						cv::Point2f(fd.boundingBox.Left, fd.boundingBox.Bottom),
						cv::Point2f(fd.boundingBox.Right, fd.boundingBox.Top)
					);
					face_container.boundingBoxIR = cv::Rect2f(
						cv::Point2f(fd.boundingBoxIR.Top, fd.boundingBoxIR.Left),
						cv::Point2f(fd.boundingBoxIR.Bottom, fd.boundingBoxIR.Right)
					);

					// rotation
					face_container.Rotation = cv::Vec4f(fd.Rotation.x, fd.Rotation.y, fd.Rotation.z, fd.Rotation.w);

					// copy face points
					std::memcpy(face_container.Points, fd.Points, FacePointType::FacePointType_Count * sizeof(PointF));
					std::memcpy(face_container.PointsIR, fd.PointsIR, FacePointType::FacePointType_Count * sizeof(PointF));

					// properties
					std::memcpy(face_container.Properties, fd.Properties, FaceProperty::FaceProperty_Count * sizeof(DetectionResult));

					// save
					mFaces.push_back(face_container);
					mUserIDs.push_back(iFace);
				}
			}

			return hr;

		}

		int GetFaceBoundingBoxesRobust(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space) const
		{

			GetFaceBoundingBoxes(bounding_boxes, space);

			float xmin, xmax, ymin, ymax, width, height;

			int srcWidth, srcHeight;
			if ((base::ImageSpace_Color & space) == base::ImageSpace_Color)
			{
				srcWidth = base::StreamSize_WidthColor;
				srcHeight = base::StreamSize_HeightColor;
			}
			else
			{
				srcWidth = base::StreamSize_WidthDepth;
				srcHeight = base::StreamSize_HeightDepth;
			}

			// check for boundary overlapping values
			for (size_t i = 0; i<bounding_boxes.size(); i++)
			{
				xmin = (bounding_boxes[i].x > 0 ? bounding_boxes[i].x : 0);
				ymin = (bounding_boxes[i].y > 0 ? bounding_boxes[i].y : 0);
				width = (bounding_boxes[i].x + bounding_boxes[i].width > (srcWidth - 1) ? srcWidth - bounding_boxes[i].x - 1 : bounding_boxes[i].width);
				height = (bounding_boxes[i].y + bounding_boxes[i].height > (srcHeight - 1) ? srcHeight - bounding_boxes[i].y - 1 : bounding_boxes[i].height);
				bounding_boxes[i].x = xmin;
				bounding_boxes[i].y = ymin;
				bounding_boxes[i].width = width;
				bounding_boxes[i].height = height;
			}
			return bounding_boxes.size();

		}


		int GetUserSceneIDs(std::vector<int> &ids) const
		{
			ids = mUserIDs;
			return mUserIDs.size();
		}

		/*
		void drawFaces(cv::Mat& dst)
		{
		for (int iFace = 0; iFace < NR_USERS; ++iFace)
		{
		cv::rectangle(
		dst,
		cv::Point(mFaces[iFace].faceBox.Bottom, mFaces[iFace].faceBox.Right),
		cv::Point(mFaces[iFace].faceBox.Bottom, mFaces[iFace].faceBox.Left),
		cv::Scalar(255, 255, 255)
		);
		}

		cv::imshow("Color image", dst);
		cv::waitKey(3);
		}

		void printFaces()
		{
		// iterate through each face reader
		for (int iFace = 0; iFace < NR_USERS; ++iFace)
		{
		if (mFaces[iFace].faceBox.Bottom > 0 && mFaces[iFace].faceBox.Top > 0)
		{
		std::cout << "Face " << iFace << " - " << mFaces[iFace].faceBox.Bottom << " - " << mFaces[iFace].faceBox.Top << "\n";
		}
		}
		}
		*/

		int GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space) const
		{
			bounding_boxes.clear();
			bool color_space = (base::ImageSpace_Color & space) == base::ImageSpace_Color;
			for (size_t j = 0; j < mFaces.size(); j++)
			{
				RectI bb;
				if (color_space) {
					bounding_boxes.push_back(mFaces[j].boundingBox);
				}
				else {
					bounding_boxes.push_back(mFaces[j].boundingBoxIR);
				}
			}
			return bounding_boxes.size();
		}

		HRESULT RenderFaceBoundingBoxes(cv::Mat &target, base::ImageSpace space) const
		{
			// get face bounding boxes
			std::vector<cv::Rect2f> bounding_boxes;
			GetFaceBoundingBoxesRobust(bounding_boxes, space);

			// draw bounding boxes
			for (size_t i = 0; i < bounding_boxes.size(); i++)
			{
				cv::rectangle(target, bounding_boxes[i], cv::Scalar(0, 0, 255), 2, cv::LINE_4);
			}

			return S_OK;
		}


		HRESULT RenderFaceFeatures(cv::Mat &target, base::ImageSpace space) const
		{
			for (size_t iFace = 0; iFace < mFaces.size(); iFace++)
			{
				for (int i = 0; i < FacePointType::FacePointType_Count; i++) {
					cv::circle(target, cv::Point2f(mFaces[iFace].Points[i].X, mFaces[iFace].Points[i].Y), 4, cv::Scalar(0, 255, 0), cv::LINE_4);
				}
			}

			return S_OK;
		}







	private:
		IKinectSensor* m_pKinectSensor;
		// tracked user ids
		std::vector<int> mUserIDs;
		std::vector<Face> mFaces;


	};

}

#endif