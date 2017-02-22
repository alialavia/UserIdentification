#ifndef FEATURES_FACE_H_
#define FEATURES_FACE_H_

// dlib
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
// misc
#include <opencv2\opencv.hpp>
#include <Config.h>
#include <tracking\FaceTracker.h>
// multithreading
#include <mutex>
#include <thread>

#define _DEBUG_FACENALIGNER

namespace features{

	// abbreviations
	typedef dlib::array2d<dlib::bgr_pixel> DlibImg;
	typedef dlib::cv_image<dlib::bgr_pixel> DlibCVMat;
	typedef std::vector<dlib::point> DlibPoints;
	typedef std::vector<cv::Point2d> CVPoints;

	/* ======================================== *\
			Kinect Face Alignment (dev.)
	\* ======================================== */

	class KinectFaceAligner : public tracking::FaceTracker {

	private:
		CVPoints mFaceLandmarksReference;
		CVPoints mMinMaxTemplate;
		cv::Rect2d mReferenceMinBB;

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
		bool AlignImage(int imgDim, cv::Mat src_bb, cv::Mat &dst) {

			// select face
			if (mFaces.size()==0) {
				std::cout << "no user!" << std::endl;
				return false;
			}

			// select face
			tracking::Face user_face = mFaces[0];

			// get rel pos of face landmarks

			std::vector<int> kinect_l_indices = { FacePointType_EyeLeft, FacePointType_EyeRight, FacePointType_Nose};
			std::vector<cv::Point2f> landmarks_in_bb = user_face.GetLandMarkPointsinBB(kinect_l_indices);


			// get template dest points for current bounding box
			int dlib_l_indices[3] = { 36, 45, 33 };	// matching dlib template indices
			std::vector<cv::Point2f> dst_points;

			// Todo: calc reference point pos in bb
			for (std::size_t i = 0; i < 3; i++)
			{
				int index = dlib_l_indices[i];
				dst_points.push_back(cv::Point2f(src_bb.cols*mMinMaxTemplate[index].x/3, src_bb.rows*mMinMaxTemplate[index].y/3));
			}

			cv::Mat warpedImg;
			cv::Mat H = cv::getAffineTransform(&landmarks_in_bb[0], &dst_points[0]);
			warpedImg = cv::Mat::zeros(src_bb.rows, src_bb.cols, src_bb.type());

			// warp image - performs hard crop from top, left
			cv::warpAffine(src_bb, warpedImg, H, warpedImg.size());

			// resize to output size
			//cv::resize(warpedImg, warpedImg, cv::Size(imgDim, imgDim));
			dst = warpedImg;
			return true;
		}

	};

	/* ======================================== *\
			Dlib Face Detector
	\* ======================================== */

	class DlibFaceDetector
	{
	protected:
		dlib::frontal_face_detector mDetector;
	public:
		DlibFaceDetector() : mDetector(dlib::get_frontal_face_detector()){
		}
		bool GetAllFaceBoundingBoxes(const cv::Mat& cvimg, std::vector<dlib::rectangle> &out);
		bool GetAllFaceBoundingBoxes(const cv::Mat& cvimg, std::vector<cv::Rect2d> &out);
		// detect faces and return larges bounding box (area)
		bool GetLargestFaceBoundingBox(const cv::Mat& cvimg, dlib::rectangle& bb, bool skip_multi = true);
	};

	/* ======================================== *\
			Threaded Dlib Face Detector
	\* ======================================== */

	class AsyncFaceDetector : public DlibFaceDetector
	{
	public:
		AsyncFaceDetector(int _MinFaceSize=80): DlibFaceDetector(), mMinFaceSize(_MinFaceSize){
		}

		void start();
		void stop();
		bool TryToDetectFaces(cv::Mat img);
		// get most recent number of face detections
		int GetNrFaces();
		// get most recent face bounding boxes
		std::vector<cv::Rect2d> GetFaces();
		
	private:
		// processing method
		void processInputImage();

		int mMinFaceSize;
		bool mRunning;
		cv::Mat mTmpImg;
		std::vector<cv::Rect2d> mFaces;
		std::mutex mLockComputation;
		std::mutex mLockAccess;
		std::thread mThread;	// processing thread
	};

	/* ======================================== *\
			Dlib Face Alignment
	\* ======================================== */

	class DlibFaceAligner : public DlibFaceDetector
	{

	private:
		CVPoints mFaceLandmarksReference;
		CVPoints mMinMaxTemplate;
		cv::Rect2d mReferenceMinBB;
		dlib::shape_predictor mShapePredictor;

	public:
		DlibFaceAligner(): DlibFaceDetector(){
			// load face model mean/reference
			LoadLandmarkReference();
		}
		void Init() {
			std::string path(PATH_MODELS);
			path += "dlib/shape_predictor_68_face_landmarks.dat";
			std::cout << "--- Loading facial landmark detector..." << std::endl;
			dlib::deserialize(path) >> mShapePredictor;
		}
		void LoadLandmarkReference();
		// detect face landmarks
		bool DetectFaceLandmarks(const cv::Mat& cvImg, const dlib::rectangle& faceBB, DlibPoints &landmarks);
		// get face landmark positions for bounding box
		std::vector<cv::Point2f> GetRefFaceLandmarkPos(const dlib::rectangle& faceBB, int indices[], int nr_indices) const;
		bool DrawFacePoints(int imgDim, const cv::Mat &src, cv::Mat &dst);
		// aling image using facial landmarks
		bool AlignImage(int imgDim, cv::Mat src, cv::Mat &dst);
	};


} // namespace

#endif
