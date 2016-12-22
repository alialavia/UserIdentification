#ifndef FEATURES_FACE_H_
#define FEATURES_FACE_H_


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include <opencv2\opencv.hpp>
#include <Config.h>

#include <tracking\FaceTracker.h>

#define _DEBUG_FACENALIGNER

namespace features{

	// abbreviations
	typedef dlib::array2d<dlib::bgr_pixel> DlibImg;
	typedef dlib::cv_image<dlib::bgr_pixel> DlibCVMat;
	typedef std::vector<dlib::point> DlibPoints;
	typedef std::vector<cv::Point2d> CVPoints;

	class KinectFaceAligner : public tracking::FaceTracker {

		CVPoints mFaceLandmarksReference;

	public:
		KinectFaceAligner(IKinectSensor* sensor) : tracking::FaceTracker(sensor) 
		{
			LoadLandmarkReference();
		}

		void LoadLandmarkReference();

		// get face landmark positions for bounding box
		CVPoints GetRefFaceLandmarkPos(const cv::Rect2d& faceBB) const;

		void DrawRefLandmarks(cv::Mat &dst, const cv::Rect2d& faceBB) {

			CVPoints pts = GetRefFaceLandmarkPos(faceBB);
			// draw
			for (size_t i = 0; i < pts.size();i++) {
				cv::circle(dst, pts[i], 1, cv::Scalar(0, 0, 255), cv::LINE_4);
			}

		}
		
		// aling image using facial landmarks
		bool AlignImage(cv::Mat &dst, int imgDim, cv::Mat &input, const cv::Rect2d& bb) {

			// reference landmarks
			CVPoints reference_pts = GetRefFaceLandmarkPos(bb);

			// calculate affine transform from three point mappings
			int ref_index[3] = { 39, 42, 33 };
			int tracker_index[3] = { FacePointType_EyeLeft, FacePointType_EyeRight, FacePointType_Nose };
			cv::Point2f alignPointsSS[3];
			cv::Point2f meanAlignPointsSS[3];

			// select face
			if (mFaces.size()==0) {
				std::cout << "no user!" << std::endl;
				return false;
			}

			// select face data
			tracking::Face target_face = mFaces[0];

			for (std::size_t i = 0; i < 3; i++)
			{
				// tracked feature location
				alignPointsSS[i].x = target_face.Points[tracker_index[i]].X;
				alignPointsSS[i].y = target_face.Points[tracker_index[i]].Y;

				// reference points
				meanAlignPointsSS[i].x = reference_pts[ref_index[i]].x;
				meanAlignPointsSS[i].y = reference_pts[ref_index[i]].y;
			}

			cv::Mat H = cv::getAffineTransform(alignPointsSS, meanAlignPointsSS);
			cv::Mat warpedImg = cv::Mat::zeros(input.rows, input.cols, input.type());
			// warp image
			cv::warpAffine(input, warpedImg, H, warpedImg.size());

			// ----------- select roi
			warpedImg = warpedImg(bb);


			dst = warpedImg;
			return true;
		}

	};

	class DlibFaceAligner
	{

	private:
		dlib::frontal_face_detector mDetector;
		dlib::shape_predictor mShapePredictor;
		CVPoints mFaceLandmarksReference;
		CVPoints mMinMaxTemplate;
		cv::Rect2d mReferenceMinBB;

	public:
		DlibFaceAligner() : mDetector(dlib::get_frontal_face_detector()) {

		}

		void Init() {
			std::string path(PATH_MODELS);
			path += "dlib/shape_predictor_68_face_landmarks.dat";

#ifdef _DEBUG_FACENALIGNER
			std::cout << "--- Loading facial landmark detector..." << std::endl;
#endif
			dlib::deserialize(path) >> mShapePredictor;
			// load face model mean/reference
			LoadLandmarkReference();
		}

		void LoadLandmarkReference();

		bool GetAllFaceBoundingBoxes(const cv::Mat& cvimg, std::vector<dlib::rectangle> &out);

		// detect faces and return larges bounding box (area)
		bool GetLargestFaceBoundingBox(const cv::Mat& cvimg, dlib::rectangle& bb, bool skip_multi = true);

		// detect face landmarks
		bool DetectFaceLandmarks(const cv::Mat& cvImg, const dlib::rectangle& faceBB, DlibPoints &landmarks);

		// get face landmark positions for bounding box
		std::vector<cv::Point2f> GetRefFaceLandmarkPos(const dlib::rectangle& faceBB, int indices[], int nr_indices) const;

		bool DrawFacePoints(int imgDim, const cv::Mat &src, cv::Mat &dst);

		// aling image using facial landmarks
		bool AlignImage(int imgDim, const cv::Mat &src, cv::Mat &dst);
	};


} // namespace

#endif
