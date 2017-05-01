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
#include <chrono>
#include <set>
#include <tuple>


#define _DEBUG_FACETRACKER

namespace tracking
{
	typedef std::vector<std::tuple<cv::Mat*, int64_t>> picture_order;

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
			a_r = (float)(interv_r_-1) / (cRMax - cRMin);
			a_p = (float)(interv_p_-1) / (cPMax - cPMin);
			a_y = (float)(interv_y_-1) / (cYMax - cYMin);
			b_r = -a_r*cRMin;
			b_p = -a_p*cPMin;
			b_y = -a_y*cYMin;
		}

		~RadialFaceGrid() {

		}
		int64_t get_timestamp()
		{
			return std::chrono::duration_cast< std::chrono::milliseconds >(
				std::chrono::system_clock::now().time_since_epoch()
				).count();
		}

		void DumpImageGrid(std::string filename = "capture", std::string log_name = "face_log.csv", std::string out_folder = "face_grid", bool append_log = false);
		void DumpImageGridInCapturingOrder(std::string filename = "capture", std::string log_name = "face_log.csv", std::string out_folder = "face_grid", bool append_log = false);
		std::vector<cv::Mat*> ExtractGrid() const;
		void ExtractGrid(std::vector<cv::Mat*> &images, std::vector<int> &weights) const;

		void ExtractUnprocessedImages(std::vector<cv::Mat*> &images, std::vector<int> &weights, bool temporal_order=true) {
				if(temporal_order)
				{
					// iterate in capturing order, check image ptr
					for (auto const& item: mImageOrder) {
						cv::Mat* mptr = std::get<0>(item);
						if (!mProcessedImages.count(mptr)) {
							cv::Vec3d angles = mAngles[mptr];
							images.push_back(mptr);
							// calc weight
							weights.push_back(CalcSampleWeight(angles[0], angles[1], angles[2]));
							mProcessedImages.insert(mptr);
						}
					}
				}else
				{
					for (auto const& target : mAngles) {
						if (!mProcessedImages.count(target.first)) {
							images.push_back(target.first);
							// calc weight
							weights.push_back(CalcSampleWeight(target.second[0], target.second[1], target.second[2]));
							mProcessedImages.insert(target.first);
						}
					}
				}

				if (images.size() > 0) {
					mLastUpdate = get_timestamp();
				}
		}

		bool ExtractUnprocessedImageBatchWithTimeout(int min_nr_images, int timeout_sec, std::vector<cv::Mat*> &images, std::vector<int> &weights, bool temporal_order=true) {

			std::vector<cv::Mat*> images_tmp;
			std::vector<int> weights_tmp;

			if (temporal_order)
			{
				// iterate in capturing order, check image ptr
				for (auto const& item : mImageOrder) {
					cv::Mat* mptr = std::get<0>(item);
					if (!mProcessedImages.count(mptr)) {
						cv::Vec3d angles = mAngles[mptr];
						images_tmp.push_back(mptr);
						// calc weight
						weights_tmp.push_back(CalcSampleWeight(angles[0], angles[1], angles[2]));
					}
				}
			}
			else
			{
				for (auto const& target : mAngles) {
					if (!mProcessedImages.count(target.first)) {
						images_tmp.push_back(target.first);
						// calc weight
						weights_tmp.push_back(CalcSampleWeight(target.second[0], target.second[1], target.second[2]));
					}
				}
			}

			if (images_tmp.size() > 0) {
				int64_t now = get_timestamp();

				// extract
				if (images_tmp.size() >= min_nr_images ||
					now - mLastExtraction > timeout_sec * 1000
					) {
					mLastExtraction = now;
					// track processed images
					for (size_t i = 0; i < images_tmp.size(); i++) {
						mProcessedImages.insert(images_tmp[i]);
					}
					images = images_tmp;
					weights = weights_tmp;
					return true;
				}
			}

			return false;
		}

		bool ExtractUnprocessedImageBatchWithTimeout(int min_nr_images, int timeout_sec, std::vector<cv::Mat*> &images, std::vector<std::tuple<int, int>> &weights, bool temporal_order = true) {

			std::vector<cv::Mat*> images_tmp;
			std::vector<std::tuple<int, int>> weights_tmp;

			if (temporal_order)
			{
				// iterate in capturing order, check image ptr
				for (auto const& item : mImageOrder) {
					cv::Mat* mptr = std::get<0>(item);
					if (!mProcessedImages.count(mptr)) {
						cv::Vec3d angles = mAngles[mptr];
						images_tmp.push_back(mptr);
						// save pitch and yaw
						weights_tmp.push_back(std::make_tuple(angles[1], angles[2]));
					}
				}
			}
			else
			{
				for (auto const& target : mAngles) {
					if (!mProcessedImages.count(target.first)) {
						images_tmp.push_back(target.first);
						// calc weight
						weights_tmp.push_back(std::make_tuple(target.second[1], target.second[2]));
					}
				}
			}

			if (images_tmp.size() > 0) {
				int64_t now = get_timestamp();

				// extract
				if (images_tmp.size() >= min_nr_images ||
					now - mLastExtraction > timeout_sec * 1000
					) {
					mLastExtraction = now;
					// track processed images
					for (size_t i = 0; i < images_tmp.size(); i++) {
						mProcessedImages.insert(images_tmp[i]);
					}
					images = images_tmp;
					weights = weights_tmp;
					return true;
				}
			}

			return false;
		}

		bool ResetIfFullOrStagnating(int max_images, int timeout_sec = 7) {

			if (nr_images() >= max_images) {
				Clear();
				return true;
			}

			// reset on timeout
			if (mLastUpdate > 0) {
				int64_t now = get_timestamp();

				// no image recorded over 7 sek
				if (now - mLastUpdate > timeout_sec*1000) {
					Clear();
					return true;
				}
			}

			return false;
		}
		
		void GetFaceGridPitchYaw(cv::Mat &dst, size_t canvas_height=500);

		static int CalcSampleWeight(int roll, int pitch, int yaw){
			// min: 0, max: 10

			// yaw
			float v_yaw = 0.;
			int w_yaw = 0;
			if (yaw <= -30) {
				v_yaw = 2.8;
			}
			else if (yaw <= 0) {
				v_yaw = 0.9333 * abs(yaw);
			}
			else if (yaw <= 30) {
				v_yaw = 0.078125 * yaw;
			}
			else {
				v_yaw = 2.5;
			}
			w_yaw = static_cast<int>(10 - v_yaw);
			
			// pitch
			float v_pitch = 0.;
			int w_pitch = 0;
			if (pitch <= -30) {
				v_pitch = 2.8;
			}
			else if (pitch <= 0) {
				v_pitch = 0.0875 * abs(pitch);
			}
			else if (pitch <= 30) {
				v_pitch = -0.00056405*pow(pitch, 3) + 0.028120486*pow(pitch, 2) - 0.013393392*pitch;
			}
			else {
				v_pitch = 10.;
			}


			w_pitch = static_cast<int>(10. - v_pitch);
			//std::cout << w_pitch << std::endl;

			int w_total = static_cast<int>(std::max(v_pitch, v_yaw));

			return w_pitch;
		}

		bool HasEnoughOrFrontalPictures(int min_nr_pictures) {
			// enough images
			if (nr_images() >= min_nr_pictures) {
				return true;
			}
			// good images
			if (frontal_images > 1) {
				return true;
			}

			return false;
		}

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
			mAngles[ptr] = ang;

			// store timestamp
			mImageOrder.push_back(std::make_tuple(ptr, get_timestamp()));

			// register frontal pictures
			if (abs(pitch) < cPFrontal && abs(yaw) < cYFrontal) {
				frontal_images++;
			}

			if (mLastUpdate == 0) {
				mLastExtraction = get_timestamp();
			}

			// save timestamp
			mLastUpdate = get_timestamp();

			return true;
		}

		void ResizeImages(int size);

		void Clear()
		{
			image_grid.Reset();
			mAngles.clear();
			mProcessedImages.clear();
			mImageOrder.clear();
			frontal_images = 0;
			mLastUpdate = 0;
		}

		size_t nr_images() {
			return mAngles.size();
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
		std::map<cv::Mat*, cv::Vec3d> mAngles;
		std::set<cv::Mat*> mProcessedImages;
		picture_order mImageOrder;
		int64_t mLastUpdate = 0;
		int64_t mLastExtraction = 0;

		size_t frontal_images = 0;

		size_t interv_r;
		size_t interv_p;
		size_t interv_y;

		// image grid resolution
		const int cRMin = -70;
		const int cRMax = 70;
		const int cPMin = -30;
		const int cPMax = 30;
		const int cYMin = -40;
		const int cYMax = 40;

		// frontal view boundaries
		const int cPFrontal = 10;
		const int cYFrontal = 10;

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
			mAngles[ptr] = ang;

			// store timestamp
			mImageOrder.push_back(std::make_tuple(ptr, get_timestamp()));

			// register frontal pictures
			if (abs(pitch) < cPFrontal && abs(yaw) < cYFrontal) {
				frontal_images++;
			}

			if (mLastUpdate == 0) {
				mLastExtraction = get_timestamp();
			}

			// save timestamp
			mLastUpdate = get_timestamp();

			// convert to grayscale
			//cv::Mat greyMat;
			//cv::cvtColor(face, greyMat, CV_BGR2GRAY);

			//if (imgproc::FocusMeasure::LAPD(greyMat) < 4) {
			//	mLabels[ptr] = Blurred;
			//}

			return true;
		}

		// overload with grid index tracking
		void GetFaceGridPitchYaw(cv::Mat &dst, size_t canvas_height) {

			size_t patch_size = std::floor(canvas_height / (float)image_grid.Size(1));
			size_t canvas_width = patch_size * image_grid.Size(2);

			mCanvasWidth = canvas_width;
			mPatchSize = patch_size;

			// allocate image
			cv::Mat canvas = cv::Mat(static_cast<int>(canvas_height), static_cast<int>(canvas_width), CV_8UC3, cv::Scalar(0, 0, 0));

			for (auto const& target : mAngles) {

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
				cv::Rect roi(iy*static_cast<int>(patch_size), ip*static_cast<int>(patch_size), canvas_copy.cols, canvas_copy.rows);
#ifdef _DEBUG_FACETRACKER
				if( roi.x < 0 || roi.width < 0 || roi.y < 0 || roi.height < 0)
				{
						std::cout << "invalid roi (<0)" << std::endl;
				}
				if (roi.x + roi.width > canvas.cols)
				{
					std::cout << "col overflow" << std::endl;

				}
				else if (roi.y + roi.height > canvas.rows)
				{
					std::cout << "row overflow" << std::endl;
				}
#endif

				canvas_copy.copyTo(canvas(roi));
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
			for (auto const& target : mAngles) {
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
		PointF Points[FacePointType::FacePointType_Count];	// facial landmarks in whole image
		PointF PointsIR[FacePointType::FacePointType_Count];
		DetectionResult Properties[FaceProperty::FaceProperty_Count];

		bool IsPhotogenic()
		{
			if(
				!Properties[FaceProperty_MouthOpen]
				&& !Properties[FaceProperty_LeftEyeClosed]
				&& !Properties[FaceProperty_RightEyeClosed]
				)
			{
				return true;
			}
			return false;
		}

		std::vector<cv::Point2f> GetLandMarkPoints(std::vector<int> landmark_indices) const{
			std::vector<cv::Point2f> points;
			for (size_t i = 0; i < landmark_indices.size();i++) {
				if (i<FacePointType::FacePointType_Count) {
					points.push_back(cv::Point2f(Points[i].X, Points[i].Y));
				}
			}
			return points;
		}

		std::vector<cv::Point2f> GetLandMarkPointsinBB(std::vector<int> landmark_indices) const{
			std::vector<cv::Point2f> points_in_frame = GetLandMarkPoints(landmark_indices);
			for (size_t i = 0; i < landmark_indices.size(); i++) {
				// calc relative position in bb
				points_in_frame[i].x -= boundingBox.x;
				points_in_frame[i].y -= boundingBox.y;
			}

			return points_in_frame;
		}

		bool IsFrontal(bool exact = false) {
			float thresh = 0.999;
			if (exact) {
				thresh = 0.9996;
			}
				
			if (Rotation[3] > thresh) {
				std::cout << Rotation[3] << std::endl;
				return true;
			}
			return false;
		}

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
			const double c_FaceRotationIncrementInDegrees = 1.0f;
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