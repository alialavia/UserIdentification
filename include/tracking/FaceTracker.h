#ifndef TRACKING_FACETRACKER_H_
#define TRACKING_FACETRACKER_H_

#include <base/UserIdentification.h>
#include <Kinect.Face.h>
#include <vector>
#include <opencv2\opencv.hpp>
#include <io/KinectInterface.h>
#include "math/Math.h"

#include "io/ImageHandler.h"

#include <opencv2\imgproc.hpp>

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

		void AllocateGrid() {
			//}
		}

		void DumpImageGrid() {
			// save the image grid to the hard drive
			std::string base_name = "face_grid";
			std::string path = "output";
			// iterate over 3d array
			for (int r = 0; r < image_grid.Size(0);r++) {
				for (int p = 0; p < image_grid.Size(1); p++) {
					for (int y = 0; y < image_grid.Size(2); y++) {
						if (!image_grid.IsFree(r, p, y)) {
							cv::Mat img = image_grid(r, p, y);
				
							// detect blur
							//cv::Mat src_gray = cv::cvtColor(img, src_gray, CV_BGR2GRAY);
							//Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);

							// write blur

							// save image
							io::ImageHandler::SaveImage(img, path, base_name + "_" + std::to_string(r) + "_" + std::to_string(p) + "_" + std::to_string(y) + ".png");
						}
					}
				}
			}
		}

		void DisplayFaceGridPitchYaw() {
			
			int canvas_height = 900;

			int patch_size = (int)((float)canvas_height / image_grid.Size(1));
			int canvas_width = patch_size * image_grid.Size(2);

			// allocate image
			cv::Mat canvas = cv::Mat(canvas_height, canvas_width, CV_8UC3, cv::Scalar(0, 0, 0));

				for (int p = 0; p < image_grid.Size(1); p++) {
					for (int y = 0; y < image_grid.Size(2); y++) {
						// take first allong rol axis
						for (int r = 0; r < image_grid.Size(0); r++) {
							if (!image_grid.IsFree(r, p, y)) {
								cv::Mat extr = image_grid(r, p, y);
								// resize

								std::cout << "patch size: " << patch_size << std::endl;
								cv::resize(extr, extr, cv::Size(patch_size, patch_size));

								// copy to left top
								extr.copyTo(canvas(cv::Rect(y*patch_size, p*patch_size, extr.cols, extr.rows)));

								// copy to
								break;
							}
						}
					}
				}
			cv::imshow("Canvas", canvas);
			cv::waitKey(3);
		}

		bool IsFree(int roll, int pitch, int yaw) {
			// check if we already got an image at this position
			int iroll = iRoll(roll);
			int	ipitch = iPitch(pitch);
			int iyaw = iYaw(yaw);
			return image_grid.IsFree(iroll, ipitch, iyaw);
		}

		// NONSAVE: check first if it is free
		bool StoreSnapshot(int roll, int pitch, int yaw, const cv::Mat &face)
		{
			int iroll = iRoll(roll);
			int	ipitch = iPitch(pitch);
			int iyaw = iYaw(yaw);
			std::cout << "Store image at: ir: " << iroll << " | ip: " << ipitch << " | iy: " << iyaw << std::endl;

			image_grid.CopyTo(iroll, ipitch, iyaw, face);
	
			return true;
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

		size_t interv_r;
		size_t interv_p;
		size_t interv_y;

		// image grid resolution
		const int cRMin = -70;
		const int cRMax = 70;
		const int cPMin = -70;
		const int cPMax = 70;
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

		std::vector<Face> GetFaces() {
			return mFaces;
		}



	private:
		IKinectSensor* m_pKinectSensor;
		// tracked user ids
		std::vector<int> mUserIDs;
		std::vector<Face> mFaces;


	};

}

#endif