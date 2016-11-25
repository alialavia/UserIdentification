#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/FaceTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>

DEFINE_string(output, "output", "Output path");

int main(int argc, char** argv)
{
	io::KinectSensorMultiSource k;


	HRESULT hr;
	cvNamedWindow("Color image", CV_WINDOW_AUTOSIZE);

	cv::Mat color_image;

	// initialize sens
	if (FAILED(k.Open())) {
		std::cout << "Initialization failed" << std::endl;
		return -1;
	}

	// skeleton tracker
	IKinectSensor* pSensor = nullptr;

	if(FAILED(k.GetSensorReference(pSensor)))
	{
		std::cout << "Sensor is not initialized" << std::endl;
		return -1;
	}

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

			// extract raw face data
			FaceData* face_data_raw = k.GetFaceDataReference();

			// copy/convert
			ft.ExtractFacialData(face_data_raw);

			// get face bounding boxes
			std::vector<cv::Rect2f> bounding_boxes;
			ft.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);

			if (bounding_boxes.size() > 0)
			{

				face_snap = color_image(bounding_boxes[0]);
				//cv::Mat face = color_image(bounding_boxes[0]);
				//// show image
				//cv::imshow("Face", face);
				//int key = cv::waitKey(3);
				//if (key == 32)	// space = save
				//{
				//	std::string filename = "face.png";
				//	std::string path = FLAGS_output;

				//	// save image
				//	io::ImageHandler::SaveImage(face, path, filename);
				//}
			}


			// faces
			std::vector<tracking::Face> faces = ft.GetFaces();
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
				yaw = -yaw;
				try
				{
					// add face if not yet capture from this angle
					if (grid.IsFree(roll, pitch, yaw)) {
						grid.StoreSnapshot(roll, pitch, yaw, face_snap);
						grid.DisplayFaceGridPitchYaw();
					}
				}
				catch (...)
				{


				}

			}

			// draw bounding boxes
			ft.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);
			ft.RenderFaceFeatures(color_image, base::ImageSpace_Color);


			// show image
			cv::imshow("Color image", color_image);
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
	grid.DumpImageGrid();


	return 0;
}
