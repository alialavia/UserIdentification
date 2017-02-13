#ifndef TRACKING_FACETRACKER_H_
#define TRACKING_FACETRACKER_H_

#include <base/UserIdentification.h>
#include <Kinect.Face.h>
#include <vector>
#include <opencv2\opencv.hpp>
#include <io/KinectInterface.h>
#include "math/Math.h"

#include "io/ImageHandler.h"
#include <io/CSVHandling.h>

#include <imgproc\ImgProcessing.h>


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

		void DumpImageGrid(std::string filename = "capture", std::string log_name = "face_log.csv", std::string out_folder = "face_grid", bool append_log = false);
		std::vector<cv::Mat*> ExtractGrid();
		void GetFaceGridPitchYaw(cv::Mat &dst, size_t canvas_height=500);

		bool IsFree(int roll, int pitch, int yaw) {

			// out of range
			if (roll<cRMin || roll > cRMax ||
				pitch<cPMin || pitch > cPMax ||
				yaw<cYMin || yaw > cYMax) {
				return false;
			}

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

		size_t nr_images() {
			return angles.size();
		}

		// ---------- index mapper
		int iRoll(int roll) {
			return static_cast<int>(floor(a_r*roll +b_r));
		}
		int iPitch(int pitch) {
			return static_cast<int>(floor(a_p*pitch + b_p));
		}
		int iYaw(int yaw) {
			return static_cast<int>(floor(a_y*yaw + b_y));
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


	class RadialFaceGridLabeled : public RadialFaceGrid {

	public:

		enum ImgLabel
		{
			None = 0,
			Blurred = 1,
			Sharp = 2,
			Unusable = 3,
		};

		RadialFaceGridLabeled(
			size_t interv_r_ = 3,
			size_t interv_p_ = 6,
			size_t interv_y_ = 10
		) : RadialFaceGrid(interv_r_, interv_p_, interv_y_)
		{


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

			// convert to grayscale
			cv::Mat greyMat;
			cv::cvtColor(face, greyMat, CV_BGR2GRAY);

			//if (imgproc::FocusMeasure::LAPD(greyMat) < 4) {
			//	mLabels[ptr] = Blurred;
			//}

			return true;
		}

		// overload with grid index tracking
		void GetFaceGridPitchYaw(cv::Mat &dst, size_t canvas_height) {

			size_t patch_size = canvas_height / image_grid.Size(1);
			size_t canvas_width = patch_size * image_grid.Size(2);

			mCanvasWidth = canvas_width;
			mPatchSize = patch_size;

			// allocate image
			cv::Mat canvas = cv::Mat(static_cast<int>(canvas_height), static_cast<int>(canvas_width), CV_8UC3, cv::Scalar(0, 0, 0));

			for (auto const& target : angles) {

				cv::Vec3d a = target.second;

				// calculate grid index
				int ir = iRoll(static_cast<int>(a[0]));
				int ip = iPitch(static_cast<int>(a[1]));
				int iy = iYaw(static_cast<int>(a[2]));

				// get image
				cv::Mat* im_ptr;
				im_ptr = target.first;

				// store grid index
				mCanvasMapPY[std::make_pair(ip, iy)] = im_ptr;

				// create copy for canvas
				cv::Mat canvas_copy = (*im_ptr).clone();
				cv::resize(canvas_copy, canvas_copy, cv::Size(static_cast<int>(patch_size), static_cast<int>(patch_size)));

				// render labels
				if (mLabels[im_ptr] != None) {

					cv::Scalar color = cv::Scalar(0, 0, 255);
					
					ImgLabel l = mLabels[im_ptr];
					std::string text;

					if (l == Blurred) {
						text = "blurred";

						cv::Mat overlay;
						canvas_copy.copyTo(overlay);
						cv::rectangle(overlay, cv::Rect(0, 0, canvas_copy.cols, canvas_copy.rows), cv::Scalar(0, 165, 255), -1);
						double alpha = 0.3;
						cv::addWeighted(overlay, alpha, canvas_copy, 1 - alpha, 0, canvas_copy);
						color = cv::Scalar(255, 255, 255);
					}
					else if (l == Unusable) {
						text = "ignore";

						cv::Mat overlay;
						canvas_copy.copyTo(overlay);
						cv::rectangle(overlay, cv::Rect(0, 0, canvas_copy.cols, canvas_copy.rows), cv::Scalar(0, 0, 255), -1);
						double alpha = 0.3;
						cv::addWeighted(overlay, alpha, canvas_copy, 1 - alpha, 0, canvas_copy);
						color = cv::Scalar(255, 255, 255);
					}
					else {
						text = std::to_string(l);
					}

					int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
					double fontScale = 2;
					int thickness = 3;
					io::ImageHandler::DrawCenteredText(text, 0.6f, cv::Point(canvas_copy.cols / 2, canvas_copy.rows / 2), canvas_copy, color);
				}

				// copy to left top
				canvas_copy.copyTo(canvas(cv::Rect(ip*static_cast<int>(patch_size), iy*static_cast<int>(patch_size), canvas_copy.cols, canvas_copy.rows)));
			}

			dst = canvas;
		}


		void SetLabelFromCanvasCoords(int cx, int cy, ImgLabel label) {
			int x = static_cast <int> (std::floor(cx / mPatchSize));
			int y = static_cast <int> (std::floor(cy / mPatchSize));

			// set label
			mLabels[mCanvasMapPY[std::make_pair(x, y)]] = label;
		}

		void SetLabel(size_t index, ImgLabel label) {
			//mLabels
		}

		void CallBackFunc(int event, int x, int y, int flags, void* userdata)
		{
			if (event == cv::EVENT_LBUTTONDOWN)
			{
				SetLabelFromCanvasCoords(x, y, Blurred);
			}
			else if (event == cv::EVENT_RBUTTONDOWN)
			{
				SetLabelFromCanvasCoords(x, y, None);
			}
			else if (event == cv::EVENT_MBUTTONDOWN)
			{
				SetLabelFromCanvasCoords(x, y, Unusable);
			}
		}

		void DumpFocusMeasuresWithLabels(std::string filename, std::string output_folder="") {

			io::CSVWriter fh(output_folder+filename);

			fh.addEntry("Label (1=Blurred | 0=Not),LAPV,LAPD,GLVN,MLAP,CEC");
			fh.EndRow();
			for (auto const& target : angles) {
				cv::Vec3d a = target.second;
				// get image
				cv::Mat* im_ptr;
				im_ptr = target.first;

				// label
				ImgLabel l = mLabels[im_ptr];
				fh.addEntry(l);

				// convert to grayscale
				cv::Mat greyMat;
				cv::cvtColor(*im_ptr, greyMat, CV_BGR2GRAY);

				// focus measures
				fh.addEntry(imgproc::FocusMeasure::LAPV(greyMat));
				fh.addEntry(imgproc::FocusMeasure::LAPD(greyMat));
				//fh.addEntry(imgproc::FocusMeasure::TENG(greyMat));
				fh.addEntry(imgproc::FocusMeasure::GLVN(greyMat));
				fh.addEntry(imgproc::FocusMeasure::MLAP(greyMat));
				fh.addEntry(imgproc::FocusMeasure::CEC(greyMat));
				fh.EndRow();
			}
		}

	private:

		std::map<std::pair<int, int>, cv::Mat*> mCanvasMapPY;
		std::map<cv::Mat*, ImgLabel> mLabels;

		// keep track of canvas properties
		size_t mCanvasWidth = 0;
		size_t mPatchSize = 0;
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

		size_t GetFaceBoundingBoxesRobust(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space) const;
		size_t GetUserSceneIDs(std::vector<int> &ids) const;
		size_t GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space) const;
		void GetFaces(std::vector<Face> &faces);
		void GetFaces(std::map<int, Face> &faces);
		void GetFaces(std::vector<Face> &faces, std::vector<int> &user_ids);
		
		HRESULT RenderFaceBoundingBoxes(cv::Mat &target, base::ImageSpace space) const;
		HRESULT RenderFaceFeatures(cv::Mat &target, base::ImageSpace space) const;

	protected:
		IKinectSensor* m_pKinectSensor;
		// tracked user ids
		std::vector<int> mUserIDs;
		std::vector<Face> mFaces;

	};

}

#endif