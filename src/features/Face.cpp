#include <features\Face.h>

using namespace features;

/* ======================================== *\
		Kinect Face Alignment (dev.)
\* ======================================== */

void KinectFaceAligner::LoadLandmarkReference()
{
	// facial landmark position for square in percent (x,y)
	std::vector<double> FACE_POINTS = {
		0.0792396913815, 0.339223741112,
		0.0829219487236, 0.456955367943,
		0.0967927109165, 0.575648016728,
		0.122141515615, 0.691921601066,
		0.168687863544, 0.800341263616,
		0.239789390707, 0.895732504778,
		0.325662452515, 0.977068762493,
		0.422318282013, 1.04329000149,
		0.531777802068, 1.06080371126,
		0.641296298053, 1.03981924107,
		0.738105872266, 0.972268833998,
		0.824444363295, 0.889624082279,
		0.894792677532, 0.792494155836,
		0.939395486253, 0.681546643421,
		0.96111933829, 0.562238253072,
		0.970579841181, 0.441758925744,
		0.971193274221, 0.322118743967,
		0.163846223133, 0.249151738053,
		0.21780354657, 0.204255863861,
		0.291299351124, 0.192367318323,
		0.367460241458, 0.203582210627,
		0.4392945113, 0.233135599851,
		0.586445962425, 0.228141644834,
		0.660152671635, 0.195923841854,
		0.737466449096, 0.182360984545,
		0.813236546239, 0.192828009114,
		0.8707571886, 0.235293377042,
		0.51534533827, 0.31863546193,
		0.516221448289, 0.396200446263,
		0.517118861835, 0.473797687758,
		0.51816430343, 0.553157797772,
		0.433701156035, 0.604054457668,
		0.475501237769, 0.62076344024,
		0.520712933176, 0.634268222208,
		0.565874114041, 0.618796581487,
		0.607054002672, 0.60157671656,
		0.252418718401, 0.331052263829,
		0.298663015648, 0.302646354002,
		0.355749724218, 0.303020650651,
		0.403718978315, 0.33867711083,
		0.352507175597, 0.349987615384,
		0.296791759886, 0.350478978225,
		0.631326076346, 0.334136672344,
		0.679073381078, 0.29645404267,
		0.73597236153, 0.294721285802,
		0.782865376271, 0.321305281656,
		0.740312274764, 0.341849376713,
		0.68499850091, 0.343734332172,
		0.353167761422, 0.746189164237,
		0.414587777921, 0.719053835073,
		0.477677654595, 0.706835892494,
		0.522732900812, 0.717092275768,
		0.569832064287, 0.705414478982,
		0.635195811927, 0.71565572516,
		0.69951672331, 0.739419187253,
		0.639447159575, 0.805236879972,
		0.576410514055, 0.835436670169,
		0.525398405766, 0.841706377792,
		0.47641545769, 0.837505914975,
		0.41379548902, 0.810045601727,
		0.380084785646, 0.749979603086,
		0.477955996282, 0.74513234612,
		0.523389793327, 0.748924302636,
		0.571057789237, 0.74332894691,
		0.672409137852, 0.744177032192,
		0.572539621444, 0.776609286626,
		0.5240106503, 0.783370783245,
		0.477561227414, 0.778476346951
	};

	double min_y = FACE_POINTS[1];
	double max_y = FACE_POINTS[1];
	double min_x = FACE_POINTS[0];
	double max_x = FACE_POINTS[0];
	for (int i = 0; i < FACE_POINTS.size(); i += 2) {
		if (min_y > FACE_POINTS[i + 1]) min_y = FACE_POINTS[i + 1];
		if (max_y < FACE_POINTS[i + 1]) max_y = FACE_POINTS[i + 1];
		if (min_x > FACE_POINTS[i]) min_x = FACE_POINTS[i];
		if (max_x < FACE_POINTS[i]) max_x = FACE_POINTS[i];
		cv::Point2d tempPoint(FACE_POINTS[i], FACE_POINTS[i + 1]);
		mFaceLandmarksReference.push_back(tempPoint);
	}

	mReferenceMinBB = cv::Rect2d(cv::Point2d(min_x, min_y), cv::Point2d(max_x, max_y));

	//MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
	for (int i = 0; i < mFaceLandmarksReference.size(); i++) {
		cv::Point2d tempPoint(
			(mFaceLandmarksReference[i].x - min_x) / (max_x - min_x),
			(mFaceLandmarksReference[i].y - min_y) / (max_y - min_y)
		);
		mMinMaxTemplate.push_back(tempPoint);
	}

}

CVPoints KinectFaceAligner::GetRefFaceLandmarkPos(const cv::Rect2d& faceBB) const {

	CVPoints res;
	for (std::size_t i = 0; i<mFaceLandmarksReference.size(); i++)
	{
		// percent
		double x = mFaceLandmarksReference[i].x, y = mFaceLandmarksReference[i].y;
		cv::Point transformedPCV(int((x * faceBB.width) + faceBB.x), int((y * faceBB.height) + faceBB.y));
		res.push_back(transformedPCV);
	}
	return res;

}

/* ======================================== *\
		Dlib Face Detector
\* ======================================== */

void DlibFaceAligner::LoadLandmarkReference()
{
	// facial landmark position for square in percent (x,y)
	std::vector<double> FACE_POINTS = {
		0.0792396913815, 0.339223741112,
		0.0829219487236, 0.456955367943,
		0.0967927109165, 0.575648016728,
		0.122141515615, 0.691921601066,
		0.168687863544, 0.800341263616,
		0.239789390707, 0.895732504778,
		0.325662452515, 0.977068762493,
		0.422318282013, 1.04329000149,
		0.531777802068, 1.06080371126,
		0.641296298053, 1.03981924107,
		0.738105872266, 0.972268833998,
		0.824444363295, 0.889624082279,
		0.894792677532, 0.792494155836,
		0.939395486253, 0.681546643421,
		0.96111933829, 0.562238253072,
		0.970579841181, 0.441758925744,
		0.971193274221, 0.322118743967,
		0.163846223133, 0.249151738053,
		0.21780354657, 0.204255863861,
		0.291299351124, 0.192367318323,
		0.367460241458, 0.203582210627,
		0.4392945113, 0.233135599851,
		0.586445962425, 0.228141644834,
		0.660152671635, 0.195923841854,
		0.737466449096, 0.182360984545,
		0.813236546239, 0.192828009114,
		0.8707571886, 0.235293377042,
		0.51534533827, 0.31863546193,
		0.516221448289, 0.396200446263,
		0.517118861835, 0.473797687758,
		0.51816430343, 0.553157797772,
		0.433701156035, 0.604054457668,
		0.475501237769, 0.62076344024,
		0.520712933176, 0.634268222208,
		0.565874114041, 0.618796581487,
		0.607054002672, 0.60157671656,
		0.252418718401, 0.331052263829,
		0.298663015648, 0.302646354002,
		0.355749724218, 0.303020650651,
		0.403718978315, 0.33867711083,
		0.352507175597, 0.349987615384,
		0.296791759886, 0.350478978225,
		0.631326076346, 0.334136672344,
		0.679073381078, 0.29645404267,
		0.73597236153, 0.294721285802,
		0.782865376271, 0.321305281656,
		0.740312274764, 0.341849376713,
		0.68499850091, 0.343734332172,
		0.353167761422, 0.746189164237,
		0.414587777921, 0.719053835073,
		0.477677654595, 0.706835892494,
		0.522732900812, 0.717092275768,
		0.569832064287, 0.705414478982,
		0.635195811927, 0.71565572516,
		0.69951672331, 0.739419187253,
		0.639447159575, 0.805236879972,
		0.576410514055, 0.835436670169,
		0.525398405766, 0.841706377792,
		0.47641545769, 0.837505914975,
		0.41379548902, 0.810045601727,
		0.380084785646, 0.749979603086,
		0.477955996282, 0.74513234612,
		0.523389793327, 0.748924302636,
		0.571057789237, 0.74332894691,
		0.672409137852, 0.744177032192,
		0.572539621444, 0.776609286626,
		0.5240106503, 0.783370783245,
		0.477561227414, 0.778476346951
	};

	double min_y = FACE_POINTS[1];
	double max_y = FACE_POINTS[1];
	double min_x = FACE_POINTS[0];
	double max_x = FACE_POINTS[0];
	for (int i = 0; i < FACE_POINTS.size(); i += 2) {
		if (min_y > FACE_POINTS[i + 1]) min_y = FACE_POINTS[i + 1];
		if (max_y < FACE_POINTS[i + 1]) max_y = FACE_POINTS[i + 1];
		if (min_x > FACE_POINTS[i]) min_x = FACE_POINTS[i];
		if (max_x < FACE_POINTS[i]) max_x = FACE_POINTS[i];
		cv::Point2d tempPoint(FACE_POINTS[i], FACE_POINTS[i + 1]);
		mFaceLandmarksReference.push_back(tempPoint);
	}

	mReferenceMinBB = cv::Rect2d(cv::Point2d(min_x, min_y), cv::Point2d(max_x, max_y));

	//MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
	for (int i = 0; i < mFaceLandmarksReference.size(); i++) {
		cv::Point2d tempPoint(
			(mFaceLandmarksReference[i].x-min_x)/(max_x - min_x),
			(mFaceLandmarksReference[i].y-min_y)/(max_y - min_y)
				);
		mMinMaxTemplate.push_back(tempPoint);
	}

}

bool DlibFaceDetector::GetAllFaceBoundingBoxes(const cv::Mat& cvimg, std::vector<dlib::rectangle> &out)
{
	std::vector<dlib::rectangle> faces;
	try
	{
		// allocate image
		dlib::array2d<dlib::rgb_pixel> dlibimg;
		dlib::assign_image(dlibimg, dlib::cv_image<dlib::bgr_pixel>(cvimg));
		faces = mDetector(dlibimg, 1);
		out = faces;
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool DlibFaceDetector::GetAllFaceBoundingBoxes(const cv::Mat& cvimg, std::vector<cv::Rect2d> &out) {

	std::vector<dlib::rectangle> faces;
	if (GetAllFaceBoundingBoxes(cvimg, faces)) {
		for (size_t i = 0; i < faces.size();i++) {
			out.push_back(cv::Rect2d(cv::Point(faces[i].left(), faces[i].top()), cv::Point(faces[i].right(), faces[i].bottom())));
		}
		return true;
	}
	return false;
}

bool DlibFaceDetector::GetLargestFaceBoundingBox(const cv::Mat& cvimg, dlib::rectangle& bb, bool skip_multi)
{
	std::vector<dlib::rectangle> bounding_boxes;

	if (!GetAllFaceBoundingBoxes(cvimg, bounding_boxes))
	{
		return false;
	}

	if (
		bounding_boxes.size() == 0 ||
		(skip_multi && bounding_boxes.size() > 1)
		)
	{
		return false;
	}

	// find largest
	dlib::rectangle largestFace;
	int largestArea = 0.;
	for (size_t i = 0; i < bounding_boxes.size(); i++) {
		int currentArea = bounding_boxes[i].width()*bounding_boxes[i].height();
		if (currentArea>largestArea)
		{
			largestArea = currentArea;
			largestFace = bounding_boxes[i];
		}
	}

#ifdef _DEBUG_FACENALIGNER
	if (largestFace.br_corner().x() > cvimg.cols || largestFace.br_corner().y() > cvimg.rows)
	{
		// overlapping bounding box - reject or adjust
		//std::cout << "INVALID FACE BOUNDING BOX - CHECK GetLargestFaceBoundingBox" << std::endl;
		//throw;
		return false;
	}
#endif

	bb = largestFace;
	return true;
}

/* ======================================== *\
		Threaded Dlib Face Detector
\* ======================================== */

void AsyncFaceDetector::start()
{
	mRunning = true;
	mThread = std::thread(&AsyncFaceDetector::processInputImage, this);
}

void AsyncFaceDetector::stop()
{
	mRunning = false;
	mThread.join();
}

bool AsyncFaceDetector::TryToDetectFaces(cv::Mat img) {
	bool succ = false;
	if (mLockComputation.try_lock()) {
		if (mTmpImg.empty()) {
			// resize (standard min detection size of Dlib is 80px)
			if (mMinFaceSize == 80) {
				mTmpImg = img.clone();
			}
			else {
				double scale_factor = mMinFaceSize / 80.;
				cv::resize(img, mTmpImg, cv::Size(img.cols * scale_factor, img.rows * scale_factor));
			}
			succ = true;
		}
		mLockComputation.unlock();
	}
	return succ;
}

int AsyncFaceDetector::GetNrFaces() {
	mLockAccess.lock();
	int nr_faces = mFaces.size();
	mLockAccess.unlock();
	return nr_faces;
}

std::vector<cv::Rect2d> AsyncFaceDetector::GetFaces() {
	std::vector<cv::Rect2d> tmp_faces;
	mLockAccess.lock();
	tmp_faces = mFaces;
	mLockAccess.unlock();
	return tmp_faces;
}

void AsyncFaceDetector::processInputImage() {
	while (mRunning) {
		mLockComputation.lock();
		if (!mTmpImg.empty()) {

			std::vector<cv::Rect2d> tmp_faces;
			//std::this_thread::sleep_for(std::chrono::milliseconds(4000));
			if (GetAllFaceBoundingBoxes(mTmpImg, tmp_faces)) {
				// success
				mFaces.clear();
				mLockAccess.lock();
				mFaces = tmp_faces;
				mLockAccess.unlock();
				mTmpImg.release();
			}

		}
		mLockComputation.unlock();
	}
}

/* ======================================== *\
		Dlib Face Alignment
\* ======================================== */

bool DlibFaceAligner::DetectFaceLandmarks(const cv::Mat& cvImg, const dlib::rectangle& faceBB, DlibPoints &landmarks)
{
	DlibPoints points;
	DlibCVMat cvdlibimg(cvImg);
	dlib::full_object_detection shape = mShapePredictor(cvdlibimg, faceBB);

	if (shape.num_parts() == 0)
	{
		return false;
	}

	for (std::size_t i = 0; i<shape.num_parts(); i++)
	{
		points.push_back(shape.part(i));
	}

	landmarks = points;
	return true;
}

// get face landmark positions for bounding box
std::vector<cv::Point2f> DlibFaceAligner::GetRefFaceLandmarkPos(const dlib::rectangle& faceBB, int indices[], int nr_indices) const {

	std::vector<cv::Point2f> res;

	for (size_t i = 0; i<nr_indices; i++)
	{

		size_t index = indices[i];

#ifdef _DEBUG_FACENALIGNER
		if (index<0 || index>mFaceLandmarksReference.size())
		{
			std::cout << "INVALID FACE LANDMARK INDEX - CHECK GetRefFaceLandmarkPos" << std::endl;
			throw;
		}
#endif
		// calculate reference position in current bounding box
		double x = mFaceLandmarksReference[index].x, y = mFaceLandmarksReference[index].y;
		// could this exceed the frame of the original image?
		cv::Point2f pixel_location((x * faceBB.width()) + faceBB.left(), (y * faceBB.height()) + faceBB.top());
		res.push_back(pixel_location);

	}
	return res;
}

bool DlibFaceAligner::DrawFacePoints(int imgDim, const cv::Mat &src, cv::Mat &dst)
{
	// extract face bounding box
	dlib::rectangle face_bounding_box;
	if (!GetLargestFaceBoundingBox(src, face_bounding_box))
	{
		return false;
	}

	// detect face landmarks
	DlibPoints face_landmarks;
	if (!DetectFaceLandmarks(src, face_bounding_box, face_landmarks))
	{
		return false;
	}

	// draw face landmarks
	dst = src;
	// calculate affine transform from three point mappings
	int landmark_indices[3] = { 39, 42, 57 };

	cv::Point2f landmark_points[3];
	std::vector<cv::Point2f> reference_points = GetRefFaceLandmarkPos(face_bounding_box, landmark_indices, 3);

	// select and convert detected landmarks points
	for (std::size_t i = 0; i < 3; i++)
	{
		landmark_points[i].x = face_landmarks[landmark_indices[i]].x();
		landmark_points[i].y = face_landmarks[landmark_indices[i]].y();
		cv::circle(dst, reference_points[i], 2, cv::Scalar(0, 255, 0), cv::LINE_4);
		cv::circle(dst, landmark_points[i], 2, cv::Scalar(0, 0, 255), cv::LINE_4);
	}
	return true;
}

// aling image using facial landmarks
bool DlibFaceAligner::AlignImage(int imgDim, cv::Mat src, cv::Mat &dst)
{

	// standard min. face size: 80px
	if(src.cols < 110)
	{
		// allows to detect faces with min 40px size
		cv::resize(src, src, cv::Size(src.cols*2, src.rows*2));

	}else if(src.cols > 200)
	{
		// downscale image for performance boost
		if(imgDim < 160)
		{
			cv::resize(src, src, cv::Size(160, 160));
		}else
		{
			// we loose some accuracy here!
		}
	}

	// extract face bounding box
	dlib::rectangle face_bounding_box;
	if (!GetLargestFaceBoundingBox(src, face_bounding_box))
	{
		return false;
	}

	// detect face landmarks
	DlibPoints face_landmarks;
	if (!DetectFaceLandmarks(src, face_bounding_box, face_landmarks))
	{
		return false;
	}

	cv::Mat warpedImg;
	if (true)
	{
		// calculate affine transform from three point mappings
		int landmark_indices[3] = { 36, 45, 33 };

		//INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
		//OUTER_EYES_AND_NOSE = [36, 45, 33]

		cv::Point2f src_points[3];
		std::vector<cv::Point2f> dst_points;
		// select and convert detected landmarks points
		for (std::size_t i = 0; i < 3; i++)
		{
			int index = landmark_indices[i];
			src_points[i].x = face_landmarks[index].x();
			src_points[i].y = face_landmarks[index].y();

			//H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices], imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
			dst_points.push_back(cv::Point2f(src.cols*mMinMaxTemplate[index].x, src.rows*mMinMaxTemplate[index].y));
		}
		// calculate transformation
		//cv::Mat H = cv::getAffineTransform(src_points, &dst_points[0]);

		cv::Mat H = cv::getAffineTransform(src_points, &dst_points[0]);
		warpedImg = cv::Mat::zeros(src.rows, src.cols, src.type());

		// warp image - performs hard crop from top, left
		cv::warpAffine(src, warpedImg, H, warpedImg.size());

		// draw points
		for (std::size_t i = 0; i < 3; i++)
		{
			//cv::circle(warpedImg, dst_points[i], 2, cv::Scalar(0, 0, 255), cv::LINE_4);
		}

		// resize to output size
		cv::resize(warpedImg, warpedImg, cv::Size(imgDim, imgDim));

	}
	else
	{

		// calculate affine transform from three point mappings
		int landmark_indices[4] = { 36, 45, 33, 8 };
		cv::Point2f src_points[4];
		std::vector<cv::Point2f> dst_points = GetRefFaceLandmarkPos(face_bounding_box, landmark_indices, 4);

		// select and convert detected landmarks points
		for (std::size_t i = 0; i < 4; i++)
		{
			src_points[i].x = face_landmarks[landmark_indices[i]].x();
			src_points[i].y = face_landmarks[landmark_indices[i]].y();
		}

		cv::Mat H = cv::getPerspectiveTransform(src_points, &dst_points[0]);
		warpedImg = cv::Mat::zeros(src.rows, src.cols, src.type());
		cv::warpPerspective(src, warpedImg, H, warpedImg.size());

		// draw points
		for (std::size_t i = 0; i < 4; i++)
		{
			//cv::circle(warpedImg, dst_points[i], 2, cv::Scalar(0, 0, 255), cv::LINE_4);
		}

		// resize to output size
		cv::resize(warpedImg, warpedImg, cv::Size(imgDim, imgDim));

	}

	// select bounding box of reference face coordinates
	//cv::Rect2f cvbb(cv::Point2f(face_bounding_box.left(), face_bounding_box.top()), cv::Point2f(face_bounding_box.right(), face_bounding_box.bottom()));

	//cv::Rect2f cvbb(
	//	cv::Point2f(
	//		mReferenceMinBB.x * face_bounding_box.width() + face_bounding_box.left(),
	//		mReferenceMinBB.y * face_bounding_box.height() + face_bounding_box.top()
	//		), 
	//	cv::Point2f(
	//		(mReferenceMinBB.x + mReferenceMinBB.width)* face_bounding_box.width() + face_bounding_box.left(),
	//		(mReferenceMinBB.y + mReferenceMinBB.height)* face_bounding_box.height() + face_bounding_box.top()
	//		)
	//);

	dst = warpedImg;
	return true;
}

