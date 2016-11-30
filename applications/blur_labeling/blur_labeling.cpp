#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/FaceTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>

DEFINE_string(output, "output", "Output path");

tracking::RadialFaceGridLabeled* g_ptr;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	g_ptr->CallBackFunc(event, x, y, flags, userdata);
}

int main(int argc, char** argv)
{
	io::KinectSensorMultiSource k;


	HRESULT hr;
	const char* cWindowLabel = "Label Selection";
	cvNamedWindow(cWindowLabel, CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback(cWindowLabel, CallBackFunc, NULL);


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

	tracking::FaceTracker ft(pSensor);
	tracking::RadialFaceGridLabeled grid(2,10,10);
	g_ptr = &grid;	// set global ptr
	cv::Mat face_snap;

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
			}

			// faces
			std::vector<tracking::Face> faces = ft.GetFaces();
			for (int i = 0; i < faces.size(); i++) {

				int roll, pitch, yaw;
				faces[i].GetEulerAngles(roll, pitch, yaw);

				try
				{
					// add face if not yet capture from this angle
					if (grid.IsFree(roll, pitch, yaw)) {
						grid.StoreSnapshot(roll, pitch, yaw, face_snap);

						//int iroll = grid.iRoll(roll);
						//int	ipitch = grid.iPitch(pitch);
						//int iyaw = grid.iYaw(yaw);
						//std::cout << "r: " << roll << " | p: " << pitch << " | y: " << yaw << "\n";
						//std::cout << "ir: " << iroll << " | ip: " << ipitch << " | iy: " << iyaw << "\n";
					}
				}
				catch (...)
				{

				}
			}

			// get face capture grid
			cv::Mat face_captures;
			grid.GetFaceGridPitchYaw(face_captures,1000);

			// draw bounding boxes
			//ft.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);
			//ft.RenderFaceFeatures(color_image, base::ImageSpace_Color);

			// show image
			cv::imshow(cWindowLabel, face_captures);

			int key = cv::waitKey(3);
			if (key == 32)	// space = save
			{
				break;
			}

		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}	// end while camera loop

	// dump faces
	//grid.DumpImageGrid();

	return 0;
}
