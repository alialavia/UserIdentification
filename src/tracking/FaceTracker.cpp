#include <tracking/FaceTracker.h>
#include "io/CSVHandling.h"
#include <ctime>


using namespace tracking;



// ---------- RadialFaceGrid

void RadialFaceGrid::DumpImageGrid(std::string img_basename, std::string log_name, std::string out_folder) {

	time_t t = std::time(0);
	long int now = static_cast<long int> (t);

	char ch = out_folder.back();
	if(ch != '/')
	{
		out_folder += "/";
	}

	io::CSVWriter o_h(out_folder+std::to_string(now)+"_"+log_name);

	// iterate over 3d array
	for (int r = 0; r < image_grid.Size(0); r++) {
		for (int p = 0; p < image_grid.Size(1); p++) {
			for (int y = 0; y < image_grid.Size(2); y++) {
				if (!image_grid.IsFree(r, p, y)) {
					cv::Mat img = image_grid(r, p, y);

					// write blur
					std::string filename = img_basename + "_" + std::to_string(r) + "_" + std::to_string(p) + "_" + std::to_string(y) + "_" + std::to_string(now) + ".png";

					// save image
					io::ImageHandler::SaveImage(img, out_folder, filename);

					// save file name and metadata
					o_h.addEntry(filename);

					cv::Vec3d precies_angles = angles[&(image_grid(r, p, y))];

					// roll, pitch, yaw
					o_h.addEntry(precies_angles[0]);
					o_h.addEntry(precies_angles[1]);
					o_h.addEntry(precies_angles[2]);
					o_h.startNewRow();
				}
			}
		}
	}
}

std::vector<cv::Mat*> RadialFaceGrid::ExtractGrid()
{
	std::vector <cv::Mat*> ptrs;
	for (auto const& target : angles) {
		ptrs.push_back(target.first);
	}
	return ptrs;
}

void RadialFaceGrid::ResizeImages(int size)
{
	for (auto const& target : angles) {
		cv::resize(*target.first, *target.first, cv::Size(size, size));
	}
}

void RadialFaceGrid::GetFaceGridPitchYaw(cv::Mat &dst, int canvas_height){


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
					bool test = image_grid.IsFree(r, p, y);

					size_t pos = image_grid.GetPos(r, p, y);
					// 75 = 0+ 1*3 + 4* 3*6

					try
					{
						cv::resize(extr, extr, cv::Size(patch_size, patch_size));
					}
					catch (...)
					{

						// resize
						std::cout << "-------------------------------\n";
						std::cout << "patch size: " << patch_size << " | " << extr.cols << " | " << extr.rows << std::endl;
					}

					// copy to left top
					extr.copyTo(canvas(cv::Rect(y*patch_size, p*patch_size, extr.cols, extr.rows)));

					// copy to
					break;
				}
			}
		}
	}

	// save
	dst = canvas;
}

// ---------- FaceTracker

HRESULT FaceTracker::ExtractFacialData(FaceData face_data[NR_USERS])
{
	HRESULT hr = E_FAIL;
	// reset tracking data
	mUserIDs.clear();
	mFaces.clear();

	for (int iFace = 0; iFace < NR_USERS; ++iFace)
	{
		FaceData fd = face_data[iFace];

		if (fd.tracked == true) {
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

int FaceTracker::GetFaceBoundingBoxesRobust(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space) const
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

int FaceTracker::GetUserSceneIDs(std::vector<int> &ids) const
{
	ids = mUserIDs;
	return mUserIDs.size();
}

int FaceTracker::GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space) const
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

HRESULT FaceTracker::RenderFaceBoundingBoxes(cv::Mat &target, base::ImageSpace space) const
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

HRESULT FaceTracker::RenderFaceFeatures(cv::Mat &target, base::ImageSpace space) const
{
	for (size_t iFace = 0; iFace < mFaces.size(); iFace++)
	{
		for (int i = 0; i < FacePointType::FacePointType_Count; i++) {
			cv::circle(target, cv::Point2f(mFaces[iFace].Points[i].X, mFaces[iFace].Points[i].Y), 4, cv::Scalar(0, 255, 0), cv::LINE_4);
		}
	}

	return S_OK;
}

std::vector<Face> FaceTracker::GetFaces() {
	return mFaces;
}