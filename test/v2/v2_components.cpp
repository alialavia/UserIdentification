#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <io/Networking.h>
#include <io/ImageHandler.h>
#include <io/RequestTypes.h>
#include <imgproc\ImgProcessing.h>
#include <io\RequestHandler.h>

typedef io::PartialImageIdentificationAligned IDReq;
typedef io::PartialUpdateAligned Update;
typedef io::PredictionFeedback PredResp;

DEFINE_int32(port, 8080, "Server port");
DEFINE_string(id_folder, "img_identification", "Image folder for identifaction captures");
DEFINE_string(update_folder, "img_update", "Image folder for update captures");


void LoadResponses(io::NetworkRequestHandler* request_handler) {

	io::NetworkRequest* request_lookup = nullptr;	// careful! the request corresponding to this pointer is already deleted!
	io::NetworkRequestType req_type;

	// ============================================= //
	// 1. handle identification responses
	// ============================================= //
	io::IdentificationResponse response;
	if (request_handler->PopResponse(&response, request_lookup))
	{
		std::cout << "--- Identification\n";
	}

	io::PredictionFeedback pred_response;
	if (request_handler->PopResponse(&pred_response, request_lookup))
	{
		std::cout << "--- Prediction Feedback\n";
	}

	io::QuadraticImageResponse img_response;
	if (request_handler->PopResponse(&img_response, request_lookup))
	{
		std::cout << "--- Image response\n";
	}

	io::OKResponse ok_response;
	if (request_handler->PopResponse(&ok_response, request_lookup))
	{
		std::cout << "--- OK\n";
	}


	io::ErrorResponse err_response;
	if (request_handler->PopResponse(&err_response, request_lookup))
	{
		std::cout << "--- Error\n";
	}
}



int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	std::cout <<
		"=============================\n"
		"Identification Images Folder: " << FLAGS_id_folder << "\n"
		"Update Images Folder: " << FLAGS_update_folder << "\n"
		"=============================\n";

	cv::Mat color_image;
	io::ImageHandler ih;


	// config to server connection
	io::TCPClient server_conn;

	server_conn.Config("127.0.0.1", FLAGS_port);
	server_conn.Connect();

	io::NetworkRequestHandler request_handler;


	// identification
	if (!ih.ChangeDirectory(FLAGS_id_folder)) {
		std::cout << "--- No images found in: " << FLAGS_id_folder << std::endl;
		return -1;
	}

	std::vector<cv::Mat> face_patches;
	std::vector<std::string> file_names;
	size_t nr_images = 0;

	// -------------------- identification

	// generate 1st model
	nr_images = ih.LoadImageBatch(face_patches, file_names, 10);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	IDReq *id_request1 = new IDReq(&server_conn, face_patches, std::vector<int>(nr_images, 5), 3);
	request_handler.addRequest(id_request1);


	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();

	// reidentify
	//nr_images = ih.LoadImageBatch(face_patches, file_names, 1);
	//imgproc::ImageProc::batchResize(face_patches, 96, 96);
	//IDReq *id_request2 = new IDReq(&server_conn, face_patches, { 6 }, 3);
	//request_handler.addRequest(id_request2);


	//cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	//cv::waitKey(0);
	//request_handler.processAllPendingRequests();


	// -------------------- updates

	if (!ih.ChangeDirectory(FLAGS_update_folder)) {
		std::cout << "--- No images found in: " << FLAGS_update_folder << std::endl;
		return -1;
	}

	// updates
	nr_images = ih.LoadImageBatch(face_patches, file_names, 2);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	Update *update1 = new Update(&server_conn, face_patches, std::vector<int>(nr_images, 6), 1);
	request_handler.addRequest(update1);

	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();

	nr_images = ih.LoadImageBatch(face_patches, file_names, 5);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	Update *update2 = new Update(&server_conn, face_patches, std::vector<int>(nr_images, 6), 1);
	request_handler.addRequest(update2);

	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();

	Update *update;

	nr_images = ih.LoadImageBatch(face_patches, file_names, 5);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	update = new Update(&server_conn, face_patches, std::vector<int>(nr_images, 6), 1);
	request_handler.addRequest(update);

	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();



	nr_images = ih.LoadImageBatch(face_patches, file_names, 5);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	update = new Update(&server_conn, face_patches, std::vector<int>(nr_images, 6), 1);
	request_handler.addRequest(update);

	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();


	nr_images = ih.LoadImageBatch(face_patches, file_names, 5);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	update = new Update(&server_conn, face_patches, std::vector<int>(nr_images, 6), 1);
	request_handler.addRequest(update);

	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();


	nr_images = ih.LoadImageBatch(face_patches, file_names, 5);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	update = new Update(&server_conn, face_patches, std::vector<int>(nr_images, 6), 1);
	request_handler.addRequest(update);

	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();


	nr_images = ih.LoadImageBatch(face_patches, file_names, 5);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	update = new Update(&server_conn, face_patches, std::vector<int>(nr_images, 6), 1);
	request_handler.addRequest(update);

	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();


	nr_images = ih.LoadImageBatch(face_patches, file_names, 5);
	imgproc::ImageProc::batchResize(face_patches, 96, 96);
	update = new Update(&server_conn, face_patches, std::vector<int>(nr_images, 6), 1);
	request_handler.addRequest(update);

	cv::imshow("bla", imgproc::ImageProc::createOne(face_patches, 1, 10));
	cv::waitKey(0);
	request_handler.processAllPendingRequests();

	server_conn.Close();
	return 0;
}
