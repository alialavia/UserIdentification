#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/FaceTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>

#include <tracking\SkeletonTracker.h>
#include <features\Face.h>

DEFINE_string(output, "output", "Output path");

int main(int argc, char** argv)
{
	io::KinectSensorMultiSource k;


	HRESULT hr;

	cv::Mat color_image;

	// initialize sens
	if (FAILED(k.Open())) {
		std::cout << "Initialization failed" << std::endl;
		return -1;
	}

	// skeleton tracker
	IKinectSensor* pSensor = nullptr;

	if (FAILED(k.GetSensorReference(pSensor)))
	{
		std::cout << "Sensor is not initialized" << std::endl;
		return -1;
	}

	tracking::SkeletonTracker st(pSensor);
	st.Init();

	// dlib aligner
	features::DlibFaceAligner dlib_aligner;
	dlib_aligner.Init();

	// kinect aligner
	features::KinectFaceAligner k_aligner(pSensor);


	tracking::FaceTracker ft(pSensor);
	tracking::RadialFaceGrid grid;
	cv::Mat face_snap;

	int cRMin = 0;
	int cRMax = 0;
	int cPMin = 0;
	int cPMax = 0;
	int cYMin = 0;
	int cYMax = 0;

	while (true) {

		// polling
		hr = k.AcquireFrame();
		if (SUCCEEDED(hr)) {

		}

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);

			// extract joint data
			IBody** bodies = k.GetBodyDataReference();
			st.ExtractJoints(bodies);

			// extract raw face data
			FaceData* face_data_raw = k.GetFaceDataReference();
			ft.ExtractFacialData(face_data_raw);



			k_aligner.ExtractFacialData(face_data_raw);

			// get face bounding boxes from face tracking
			//std::vector<cv::Rect2f> bounding_boxes;
			//ft.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);

			// get face bounding boxes from skeleton tracking
			std::vector<cv::Rect2f> bounding_boxes;
			std::vector<int> user_ids;
			st.GetFaceBoundingBoxesRobust(bounding_boxes, user_ids, base::ImageSpace_Color);

			if (bounding_boxes.size() > 0)
			{

				face_snap = color_image(bounding_boxes[0]);

				// kinect aligner
				if (false) {
					k_aligner.DrawRefLandmarks(color_image, bounding_boxes[0]);
					cv::Mat aligned;
					bool succ = k_aligner.AlignImage(aligned, 200, color_image, bounding_boxes[0]);
					if (succ) {
						cv::imshow("Face", aligned);
						int key = cv::waitKey(3);
					}
				}

				// dlib aligner
				if (true) {
					try
					{
						dlib::rectangle bb;
						cv::Mat aligned;
						if (
							dlib_aligner.GetLargestFaceBoundingBox(face_snap, bb) && 
							dlib_aligner.AlignImage(200, face_snap, aligned, bb)
							) {
							cv::Rect2f bb_cv(cv::Point2f(bb.top(), bb.left()), cv::Point2f(bb.bottom(), bb.right()));
							// show image
							//cv::imshow("Face", aligned(bb_cv));
							cv::imshow("Face", aligned);
							int key = cv::waitKey(3);
						}
					}
					catch (std::exception e)
					{
						std::cout << "Failed to process image. Caught exception " << e.what() << std::endl;
					}
				}


			}	// endif faces detected


			// faces
			std::vector<tracking::Face> faces;
			ft.GetFaces(faces);
			for (int i = 0; i < faces.size();i++) {

				int roll, pitch, yaw;
				faces[i].GetEulerAngles(roll, pitch, yaw);

				if (roll < cRMin) {cRMin = roll;}
				if (roll > cRMax) {cRMax = roll;}
				if (pitch < cPMin) { cPMin = pitch; }
				if (pitch > cPMax) { cPMax = pitch; }
				if (yaw < cYMin) { cYMin = yaw; }
				if (yaw > cYMax) { cYMax = yaw; }

				//std::cout << "r: " << roll << " | p: " << pitch << " | y: " << yaw << std::endl;
				// mirror yaw
				//yaw = -yaw;

				try
				{
					// add face if not yet capture from this angle
					if (grid.IsFree(roll, pitch, yaw)) {
						//grid.StoreSnapshot(roll, pitch, yaw, face_snap);
					}
				}
				catch (...)
				{


				}
			}

			//// get face capture grid
			//cv::Mat face_captures;
			//grid.GetFaceGridPitchYaw(face_captures);

			// draw bounding boxes
			//ft.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);
			ft.RenderFaceFeatures(color_image, base::ImageSpace_Color);

			// show image
			cv::imshow("Faces", color_image);
			int key = cv::waitKey(3);
			if (key == 32)	// space = save
			{
				break;
			}

		} else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}	// end while camera loop


	std::cout << "R: " << cRMin << "° .. " << cRMax << "°" << std::endl;
	std::cout << "P: " << cPMin << "° .. " << cPMax << "°" << std::endl;
	std::cout << "Y: " << cYMin << "° .. " << cYMax << "°" << std::endl;

	// dump faces
	//grid.DumpImageGrid();


	return 0;
}
