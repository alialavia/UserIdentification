#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <io/Networking.h>
#include <io/ImageHandler.h>
#include <io/RequestTypes.h>

typedef io::ImageIdentificationAligned IDReq;
typedef io::PredictionFeedback PredResp;

DEFINE_int32(port, 8080, "Server port");
DEFINE_string(id_folder, "img_identification", "Image folder for identifaction captures");
DEFINE_string(update_folder, "img_update", "Image folder for update captures");


void LoadResponses(io::TCPClient* server_conn) {

	// wait for reponse
	io::IdentificationResponse response(server_conn);

	if (!response.Load(&response_code)) {
		std::cout << "--- An error occurred during identification: ResponseType " << response_code << std::endl;
		server_conn->Close();
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
	int response_code = 0;

	// config to server connection
	io::TCPClient server_conn;

	server_conn.Config("127.0.0.1", FLAGS_port);
	server_conn.Connect();

	// identification
	if (!ih.ChangeDirectory(FLAGS_id_folder)) {
		std::cout << "--- No images found in: " << FLAGS_id_folder << std::endl;
		return -1;
	}

	std::vector<cv::Mat> face_patches;
	std::vector<std::string> file_names;
	size_t nr_images = 0;
	nr_images = ih.LoadImageBatch(face_patches, file_names, 5);


	// generate request
	std::cout << "--- Request Identification" << std::endl;

	if (nr_images>0) {
		cv::imshow("test", face_patches[0]);
		cv::waitKey(0);
	}


	IDReq id_request(&server_conn, face_patches);
	id_request.SubmitRequest();

	std::cout << "--- Request submited. Waiting for response" << std::endl;

	LoadResponses(&server_conn);





	//std::cout << "--- Got response. Closing connection." << std::endl;

	//server_conn.Close();

	//int user_id = response.mUserID;
	//std::cout << "--- Identified User ID " << user_id << std::endl;

	server_conn.Close();
	return 0;
}