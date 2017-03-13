#include <iostream>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include <gflags/gflags.h>
#include "io/CSVHandling.h"
#include <chrono>
#include <features/Face.h>
#include <io/ImageHandler.h>



DEFINE_string(log_name, "alignment_timings.csv", "Output log (e.g. alignment_timings.csv)");
DEFINE_string(dir, "", "Image folder)");
DEFINE_int32(iterations, 1, "Number of iterations");

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	features::DlibFaceAligner aligner;
	aligner.Init();

	io::ImageHandler ih;
	io::CSVWriter csv(FLAGS_log_name);
	
	std::vector<int> timings_ms;
	cv::Mat dst;
	
	std::vector<std::string> filenames;
	//ih.GetFilesInDirectory(filenames, FLAGS_dir);

	//if (filenames.size() == 0) {
	//	std::cout << "No files found in directory!" << std::endl;
	//	return 0;
	//}
	//else {
	//	std::cout << "Processing " << filenames.size() << " images\n";
	//}

	// load images
	std::vector<cv::Mat> img_batch;
	int nr_processed = 0;

	ih.ChangeDirectory(FLAGS_dir);

	while (ih.LoadImageBatch(img_batch, filenames, 10) > 0) {

		nr_processed += img_batch.size();

		// align images one by one
		for (size_t i = 0; i < img_batch.size();i++) {
			timings_ms.clear();

			// iterate
			for (int iter = 0; iter < FLAGS_iterations; iter++) {
				
				auto t1 = std::chrono::high_resolution_clock::now();
				aligner.AlignImage(96, img_batch[i], dst);
				auto t2 = std::chrono::high_resolution_clock::now();
				timings_ms.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
			}

			// 1 row per image
			csv.addRow(timings_ms);
		}
		std::cout << "Processed: " << nr_processed << std::endl;
	}


	return 0;
}
