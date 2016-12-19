#include <features\Face.h>

using namespace features;

void DlibFaceAligner::detect_faces(cv::Mat &cv_mat, cv::Mat roi) {

	// opencv to dlib
	dlib::array2d<dlib::rgb_pixel> img;
	dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(roi));

	// upsample image
	//dlib::pyramid_up(img);

	std::vector<dlib::rectangle> dets = mDetector(img);
	std::cout << "---- Number of faces detected: " << dets.size() << std::endl;
	std::vector<cv::Point> face_centers;

	//Bounding Box Centroid
	for (size_t i = 0; i < dets.size(); i++) {
		cv::Point center = cv::Point((dets[i].right() - dets[i].left()) / 2, (dets[i].bottom() - dets[i].top()) / 2);
		face_centers.push_back(center);
		// draw face bounding box
		cv::rectangle(cv_mat, cv::Point(dets[i].left(), dets[i].top()), cv::Point(dets[i].right(), dets[i].bottom()), cv::Scalar(0, 0, 255), 4);
	}
}

dlib::rectangle DlibFaceAligner::GetLargestFaceBoundingBox(const cv::Mat& cvimg)
{
	dlib::array2d<dlib::rgb_pixel> img;
	dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(cvimg));
	std::vector<dlib::rectangle> faces = mDetector(img);
	dlib::rectangle largestFace;
	int largestArea = 0;
	for (dlib::rectangle face : faces) {
		int currentArea = face.width()*face.height();
		if (currentArea>largestArea)
		{
			largestArea = currentArea;
			largestFace = face;
		}
	}
	return largestFace;
}

DlibPoints DlibFaceAligner::DetectFaceLandmarks(const cv::Mat& cvImg, const dlib::rectangle& faceBB)
{
	DlibPoints landmarks;
	DlibCVMat cvdlibimg(cvImg);
	dlib::full_object_detection shape = mShapePredictor(cvdlibimg, faceBB);
	for (std::size_t i = 0; i<shape.num_parts(); i++)
	{
		landmarks.push_back(shape.part(i));
	}

	return landmarks;
}

DlibPoints DlibFaceAligner::GetRefFaceLandmarkPos(const dlib::rectangle& faceBB) const
{

#ifdef _DEBUG_FACENALIGNER
	if (mFaceLandmarksReference.size() == 0)
	{
		throw std::range_error("Face Aligner was not initialized. Reference point positions are missing.");
	}
#endif

	DlibPoints res;
	for (std::size_t i = 0; i<mFaceLandmarksReference.size(); i++)
	{
		double x = mFaceLandmarksReference[i].x, y = mFaceLandmarksReference[i].y;
		dlib::point transformedP(int((x * faceBB.width()) + faceBB.left()),
			int((y * faceBB.height()) + faceBB.top()));
		res.push_back(transformedP);
	}
	return res;
}

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

	for (int i = 0; i < FACE_POINTS.size(); i += 2) {
		cv::Point2d tempPoint(FACE_POINTS[i], FACE_POINTS[i + 1]);
		mFaceLandmarksReference.push_back(tempPoint);
	}
}

cv::Mat DlibFaceAligner::AlignImage(int imgDim, cv::Mat &input, const dlib::rectangle& bb)
{

	// convert opencv image to dlib mat
	DlibImg img;
	dlib::assign_image(img, DlibCVMat(input));
	auto dlibImb = dlib::toMat(img);

	DlibPoints alignPoints = DetectFaceLandmarks(dlibImb, bb);
	DlibPoints meanAlignPoints = GetRefFaceLandmarkPos(bb);

	int left = meanAlignPoints[0].x(), top = meanAlignPoints[0].y(),
		right = meanAlignPoints[0].x(), bottom = meanAlignPoints[0].y();

	for (std::size_t i = 0; i < meanAlignPoints.size(); i++)
	{
		left = std::min<long>(left, meanAlignPoints[i].x());
		top = std::min<long>(top, meanAlignPoints[i].y());
		right = std::max<long>(right, meanAlignPoints[i].y());
		bottom = std::max<long>(bottom, meanAlignPoints[i].y());
	}

	// calc bounding box for reference face features
	dlib::rectangle tightBb(left, top, right, bottom);

	bool warp_affine = true;
	cv::Mat warpedImg;

	if (warp_affine) {
		// calculate affine transform from three point mappings
		int ss[3] = { 39, 42, 57 };
		cv::Point2f alignPointsSS[3];
		cv::Point2f meanAlignPointsSS[3];
		for (std::size_t i = 0; i < 3; i++)
		{
			alignPointsSS[i].x = alignPoints[ss[i]].x();
			alignPointsSS[i].y = alignPoints[ss[i]].y();
			meanAlignPointsSS[i].x = meanAlignPoints[ss[i]].x();
			meanAlignPointsSS[i].y = meanAlignPoints[ss[i]].y();
		}

		cv::Mat H = cv::getAffineTransform(alignPointsSS, meanAlignPointsSS);
		cv::Mat cvImg = dlib::toMat(img);
		warpedImg = cv::Mat::zeros(cvImg.rows, cvImg.cols, cvImg.type());
		// warp image
		cv::warpAffine(cvImg, warpedImg, H, warpedImg.size());
	}
	else {
		// calculuate perspective transform from four point mappings
		int ss[4] = { 36, 45, 48, 54 };
		cv::Point2f alignPointsSS[4];
		cv::Point2f meanAlignPointsSS[4];
		for (std::size_t i = 0; i < 4; i++)
		{
			alignPointsSS[i].x = alignPoints[ss[i]].x();
			alignPointsSS[i].y = alignPoints[ss[i]].y();
			meanAlignPointsSS[i].x = meanAlignPoints[ss[i]].x();
			meanAlignPointsSS[i].y = meanAlignPoints[ss[i]].y();
		}

		cv::Mat H = cv::getPerspectiveTransform(alignPointsSS, meanAlignPointsSS);
		cv::Mat cvImg = dlib::toMat(img);
		warpedImg = cv::Mat::zeros(cvImg.rows, cvImg.cols, cvImg.type());
		cv::warpPerspective(cvImg, warpedImg, H, warpedImg.size());

		cv::imshow("Face2", warpedImg);
		int key = cv::waitKey(3);
	}

	// extract face bounding box in warped image
	dlib::rectangle wBb = GetLargestFaceBoundingBox(warpedImg);

	if (wBb.width() <= 0 || wBb.height() <= 0)
	{
		std::cout << "Error with bounding box." << std::endl;
		throw std::invalid_argument("Error with bounding box.");
	}

	DlibPoints wAlignPoints = DetectFaceLandmarks(warpedImg, wBb);
	DlibPoints wMeanAlignPoints = GetRefFaceLandmarkPos(wBb);

	if (warpedImg.channels() != 3)
	{
		throw std::invalid_argument("Image does not have 3 channels.");
	}

	left = wAlignPoints[0].x(), top = wAlignPoints[0].y(),
		right = wAlignPoints[0].x(), bottom = wAlignPoints[0].y();

	for (std::size_t i = 0; i < wAlignPoints.size(); i++)
	{
		left = std::min<long>(left, wAlignPoints[i].x());
		top = std::min<long>(top, wAlignPoints[i].y());
		right = std::max<long>(right, wAlignPoints[i].y());
		bottom = std::max<long>(bottom, wAlignPoints[i].y());
	}

	int w = warpedImg.size[0], h = warpedImg.size[1];

	cv::Rect wrect(
		std::max<std::size_t>(wBb.left(), 0),
		std::max<std::size_t>(wBb.top(), 0),
		std::min<std::size_t>(wBb.width(), w),
		std::min<std::size_t>(wBb.height(), h));

	//cv::rectangle(warpedImg, wrect, cv::Scalar(0, 0, 255));
	//cv::resize(warpedImg, warpedImg, cv::Size(imgDim, imgDim));
	//return warpedImg;
	// return bb

	cv::resize(warpedImg(wrect), warpedImg, cv::Size(imgDim, imgDim));

	return warpedImg;

}
