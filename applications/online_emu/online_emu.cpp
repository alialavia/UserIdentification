#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <io/Networking.h>
#include <io/ImageHandler.h>
#include <io/RequestTypes.h>

typedef io::ImageIdentification IDReq;

DEFINE_int32(port, 8080, "Server port");
DEFINE_string(id_folder, "img_identification", "Image folder for identifaction captures");
DEFINE_string(update_folder, "img_update", "Image folder for update captures");

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

	// identification
	if (!ih.ChangeDirectory(FLAGS_id_folder)) {
		std::cout << "--- No images found in: " << FLAGS_id_folder << std::endl;
		return -1;
	}

	std::vector<cv::Mat> face_patches;
	std::vector<std::string> file_names;
	ih.LoadImageBatch(face_patches, file_names, 5);

	// generate request
	std::cout << "--- Request Identification" << std::endl;
	IDReq id_request(&server_conn, face_patches);
	server_conn.Connect();
	id_request.SubmitRequest();

	std::cout << "--- Request submited. Waiting for response" << std::endl;

	// wait for reponse
	io::IdentificationResponse response(&server_conn);

	if (!response.Load(&response_code)) {
		std::cout << "--- An error occurred during identification: ResponseType " << response_code << std::endl;
		server_conn.Close();
		return -1;
	}

	std::cout << "--- Got response. Closing connection." << std::endl;

	server_conn.Close();

	int user_id = response.mUserID;
	std::cout << "--- Identified User ID " << user_id << std::endl;
	
	// updates
	if (!ih.ChangeDirectory(FLAGS_update_folder)) {
		std::cout << "--- No images found in: " << FLAGS_update_folder << std::endl;
		return -1;
	}

	for (int i=0;i<2;i++) 
	{

		// load 2 images
		ih.LoadImageBatch(face_patches, file_names, 2);

		// generate request
		server_conn.Connect();
		io::EmbeddingCollectionByIDAligned update_request(&server_conn, face_patches, user_id);
		update_request.SubmitRequest();

		// wait for reponse
		io::OKResponse ok_response(&server_conn);
		if (!ok_response.Load(&response_code)) {
			std::cout << "--- An error occurred during update: ResponseType " << response_code << " | expected: " << ok_response.cTypeID << std::endl;
			server_conn.Close();
			return -1;
		}

		

	}

	server_conn.Close();
	std::cout << "--- Updates finished\n";

	return 0;
}
