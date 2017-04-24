#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>
#include <user/User.h>
#include <tracking\FaceTracker.h>


void put_img(cv::Mat img, tracking::RadialFaceGrid* pGrid, int roll, int pitch, int yaw) {


	if (pGrid->IsFree(roll, pitch, yaw))
	{
			try
			{
				pGrid->StoreSnapshot(roll, pitch, yaw, img);
			}
			catch (...)
			{
			}
	}
}

int main(int argc, char** argv)
{

	tracking::RadialFaceGrid grid(2, 6, 7);

	cv::Mat M1(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
	cv::Mat M2(100, 100, CV_8UC3, cv::Scalar(0, 255, 0));
	cv::Mat M3(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
	cv::Mat M4(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat M5(100, 100, CV_8UC3, cv::Scalar(255, 255, 0));
	cv::Mat M6(100, 100, CV_8UC3, cv::Scalar(50, 0, 50));
	cv::Mat M7(100, 100, CV_8UC3, cv::Scalar(100, 50, 0));

	cv::imshow("bla", imgproc::ImageProc::createOne({M1, M2, M3, M4, M5, M6}, 1, 10));
	cv::waitKey(0);
	cv::destroyAllWindows();


	put_img(M1, &grid, 0, 0, 0);
	put_img(M2, &grid, 20, 0, 0);
	put_img(M3, &grid, 0, 20, 0);



	std::vector<cv::Mat*> face_patches;
	std::vector<int> sample_weights;
	bool has_samples;

	has_samples = grid.ExtractUnprocessedImageBatchWithTimeout(2, 5, face_patches, sample_weights);
	if (has_samples) {
		cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
		cv::waitKey(0);
		cv::destroyAllWindows();
	}


	put_img(M4, &grid, 0, 0, 20);
	put_img(M5, &grid, 20, 20, 0);
	put_img(M6, &grid, 0, 20, 20);

	has_samples = grid.ExtractUnprocessedImageBatchWithTimeout(2, 5, face_patches, sample_weights);
	if (has_samples) {
		cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	return 0;
} 
