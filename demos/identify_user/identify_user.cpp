#include <iostream>
#include <io/KinectInterface.h>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include "io/ImageHandler.h"
#include <gflags/gflags.h>
#include <io/Networking.h>
#include <io/ImageHandler.h>
#include <io/RequestTypes.h>
#include <imgproc/ImgProcessing.h>

DEFINE_int32(port, 8080, "Server port");
DEFINE_string(ip, "127.0.0.1", "Server ip");
DEFINE_string(picture, "", "Face photo");
DEFINE_string(img_folder, "", "Input image folder");


bool identify_user(io::TCPClient *pServerConn, std::vector<cv::Mat> face_photos) {


	// face detection and alignment


	io::ImageIdentification req(pServerConn, face_photos);
	req.SubmitRequest();

	std::cout << "--- Request submited. Waiting for response" << std::endl;

	// wait for reponse
	io::IdentificationResponse response(pServerConn);
	int response_code = 0;
	if (!response.Load(&response_code)) {
		std::cout << "--- An error occurred during identification: ResponseType " << response_code << std::endl;
		return false;
	}


	// Response:
	//response.mImage;
	//response.mUserID;
	//response.mUserNiceName;
	//response.mConfidence;

	std::cout << "\n____________________________\nIdentified user:\n";
	std::cout << "ID: " << response.mUserID << " | nicename: " << response.mUserNiceName << " | confidence: " << response.mConfidence << "\n";

	if (!response.mImage.empty())
	{
		cv::imshow("Profile Pictures", response.mImage);
		cv::waitKey(0);
	}
	else
	{
		std::cout << "No profile pictures taken yet...\n";
	}
	return true;
}


int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (FLAGS_picture == "") {
		std::cout << "Please provide a photo for identification\n";
		return -1;
	}

	io::ImageHandler ih;
	std::vector<cv::Mat> photos;
	cv::Mat img_tmp;
	ih.ChangeDirectory(FLAGS_img_folder);

	img_tmp = cv::imread(FLAGS_picture, CV_LOAD_IMAGE_COLOR);   // Read the file
	if (!img_tmp.data) {
		std::cout << "Picture could not be loaded...\n";
		return -1;
	}
	photos.push_back(img_tmp);

	// config to server connection
	io::TCPClient server_conn;
	server_conn.Config(&FLAGS_ip[0], FLAGS_port);
	server_conn.Connect();
	identify_user(&server_conn, photos);
	server_conn.Close();

	return 0;
}
