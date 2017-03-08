#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "tracking/FaceTracker.h"
#include "io/ImageHandler.h"
#include <gflags/gflags.h>


DEFINE_string(stat_file, "labeled_focus_measures.csv", "Statistics file name");
DEFINE_string(output_folder, "pictures", "Output path");
DEFINE_string(img_basename, "picture", "Image basename");
DEFINE_string(lock_axis, "", "Axis to lock: {roll, pitch, yaw}. Only takes picture if this axis is near zero (+-1°)");
DEFINE_bool(subtract_bg, false, "Save a copy of the images with subtracted background");

tracking::RadialFaceGridLabeled* g_ptr;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	g_ptr->CallBackFunc(event, x, y, flags, userdata);
}

enum State
{
	State_none = 0,
	State_capturing = 1,
	State_dump_blur_stats = 2,
	State_dump_images = 3
};

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

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

	//if(FLAGS_lock_axis == "roll")
	//{
	//	res_roll = 1;
	//}else if (FLAGS_lock_axis == "pitch")
	//{
	//	res_pitch = 1;
	//}else if (FLAGS_lock_axis == "yaw")
	//{
	//	res_yaw = 1;
	//}

	tracking::RadialFaceGridLabeled grid(5,10,20);
	g_ptr = &grid;	// set global ptr
	cv::Mat face_snap;
	enum State STATE = State_none;

	// print instructions
	std::cout << "=====================================\n"
		"          INSTRUCTIONS\n"
		"=====================================\n"
		"[1]: autocollect face snapshots. When finished use:\n"
		"		[s]: save blur statistics\n"
		"		[i]: save images\n"
		"[2]: autocollect face snapshots (unlimited, savepoint triggered in intervals)\n"
		"[q]: Quit\n"
		"--------------------------------------\n"
		"During picture collection use [LMB], [RMB], [MMB] to flag, reset or ignore individual pictures\n"
		"\n\n";

	cv::destroyAllWindows();
	int key = (int)('-1');
	int key_save = (int)('-1');

	while (true) {

		// polling
		hr = k.AcquireFrame();

		// check if there is a new frame available
		if (SUCCEEDED(hr)) {

			// get color image
			k.GetImageCopyRGB(color_image);


			// mode selection
			if (STATE == State_none)
			{
				if ((int)('1') == key || (int)('2') == key)	// space = save
				{
					STATE = State_capturing;
					std::cout << "--- Start image capturing...\n";
				}
				else if ((int)('q') == key)
				{
					std::cout << "--- Terminating...\n";
					break;
				}
			}

			// face grid
			if (STATE == State_capturing)
			{

				// extract raw face data
				FaceData* face_data_raw = k.GetFaceDataReference();

				// copy/convert
				ft.ExtractFacialData(face_data_raw);

				// get face bounding boxes
				std::vector<cv::Rect2f> bounding_boxes;
				ft.GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);


				if (bounding_boxes.size() > 0)
				{

					cv::Rect2d bb = bounding_boxes[0];
		/*			bb.x -= 10;
					bb.y -= 10;
					bb.width += 10;
					bb.height += 10;*/

					face_snap = color_image(bb);
				}

				// get faces
				std::vector<tracking::Face> faces;
				std::vector<int> face_ids;
				ft.GetFaces(faces, face_ids);
				for (int i = 0; i < faces.size(); i++) {

					int roll, pitch, yaw;
					faces[i].GetEulerAngles(roll, pitch, yaw);


					if (
						FLAGS_lock_axis == "roll" &&
						(roll > 5 || roll < -5)
						)
					{
						std::cout << "roll: " << roll << std::endl;
						continue;
					}
					if (
						FLAGS_lock_axis == "pitch" &&
						(pitch > 5 || pitch < -5)
						)
					{
						std::cout << "pitch: " << pitch << std::endl;
						continue;
					}
					if (
						FLAGS_lock_axis == "yaw" &&
						(yaw > 5 || yaw < -5)
						)
					{
						std::cout << "yaw: " << yaw << std::endl;
						continue;
					}

					// fix roll axis
					if(FLAGS_lock_axis == "yaw" || FLAGS_lock_axis == "pitch")
					{
						if (
							(roll > 5 || roll < -5)
							)
						{
							continue;
						}
					}

					try
					{
						// add face if not yet capture from this angle
						if (grid.IsFree(roll, pitch, yaw)) {

							// convert to grayscale
							cv::Mat greyMat;
							cv::cvtColor(face_snap, greyMat, CV_BGR2GRAY);

							grid.StoreSnapshot(roll, pitch, yaw, face_snap);

							if (imgproc::FocusMeasure::LAPD(greyMat) > 4) {

							}

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
				grid.GetFaceGridPitchYaw(face_captures, 1000);

				// draw bounding boxes
				//ft.RenderFaceBoundingBoxes(color_image, base::ImageSpace_Color);
				//ft.RenderFaceFeatures(color_image, base::ImageSpace_Color);

				// show image
				cv::imshow(cWindowLabel, face_captures);
				key_save = cv::waitKey(5);

				if((int)('1') == key){
					if ((int)('s') == key_save)	// space = save
					{
						std::cout << "--- Saving blur metrics...\n";
						grid.DumpFocusMeasuresWithLabels(FLAGS_stat_file, FLAGS_output_folder);
						grid.Clear();
						STATE = State_none;
						cv::destroyAllWindows();
					}
					else if ((int)('i') == key_save)
					{
						std::cout << "--- Saving images...\n";
						grid.DumpImageGrid(FLAGS_img_basename, "picture_log.csv", FLAGS_output_folder);
						grid.Clear();
						STATE = State_none;
						cv::destroyAllWindows();
					}
				}
				else if ((int)('2') == key) {
					// autosave
					if (grid.nr_images() > 5) {
						std::cout << "Autosaving..." << std::endl;
						grid.DumpImageGrid(FLAGS_img_basename, "picture_log.csv", FLAGS_output_folder, true);
						grid.Clear();
						cv::destroyAllWindows();
					}
					if ((int)('q') == key_save)
					{
						std::cout << "--- Terminating...\n";
						break;
					}
				}
			}

			// wait for input
			if (STATE == State_none)
			{
				cv::imshow("Camera", color_image);
				key = cv::waitKey(3);
			}

		}
		else {
			// error handling (e.g. check if serious crash or just pending frame in case our system runs > 30fps)

		}
	}	// end while camera loop


	return 0;
}
