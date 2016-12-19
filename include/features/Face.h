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

#define _DEBUG_FACENALIGNER

namespace features{

	// abbreviations
	typedef dlib::array2d<dlib::bgr_pixel> DlibImg;
	typedef dlib::cv_image<dlib::bgr_pixel> DlibCVMat;
	typedef std::vector<dlib::point> DlibPoints;
	typedef std::vector<cv::Point2d> CVPoints;

	class DlibFaceAligner
	{

	private:
		dlib::frontal_face_detector mDetector;
		dlib::shape_predictor mShapePredictor;
		CVPoints mFaceLandmarksReference;

	public:
		DlibFaceAligner() : mDetector(dlib::get_frontal_face_detector()){
			
		}

		void Init() {
			std::string path(PATH_MODELS);
			path += "dlib/shape_predictor_68_face_landmarks.dat";
			dlib::deserialize(path) >> mShapePredictor;
			// load face model mean/reference
			LoadLandmarkReference();
		}

		void LoadLandmarkReference();

		// detect images in roi
		void detect_faces(cv::Mat &cv_mat, cv::Mat roi);

		// detect faces and return larges bounding box (area)
		dlib::rectangle GetLargestFaceBoundingBox(const cv::Mat& cvimg);

		// detect face landmarks
		DlibPoints DetectFaceLandmarks(const cv::Mat& cvImg, const dlib::rectangle& faceBB);
	
		// get face landmark positions for bounding box
		DlibPoints GetRefFaceLandmarkPos(const dlib::rectangle& faceBB) const;

		// aling image using facial landmarks
		cv::Mat AlignImage(int imgDim, cv::Mat &input, const dlib::rectangle& bb);

	};

} // namespace

#endif
