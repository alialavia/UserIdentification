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


bool get_profile_pictures(io::TCPClient *pServerConn) {

	// request images from server
	io::GetProfilePictures req(pServerConn);
	req.SubmitRequest();

	std::cout << "--- Request submited. Waiting for response" << std::endl;

	// wait for reponse
	io::ProfilePictures response(pServerConn);
	int response_code = 0;

	if (!response.Load(&response_code)) {
		std::cout << "--- An error occurred during identification: ResponseType " << response_code << std::endl;
		return false;
	}

	std::vector<int> user_ids;
	std::vector<cv::Mat> profile_pics;

	// load images
	for (size_t i = 0; i < response.mUserIDs.size(); i++) {
		user_ids.push_back(response.mUserIDs[i]);
		profile_pics.push_back(response.mImages[i]);
	}

	// write id on profile pictures
	for (size_t i = 0; i<profile_pics.size(); i++)
	{
		cv::Rect bg_patch = cv::Rect(0, 0, 30, 20);
		profile_pics[i](bg_patch) = cv::Scalar(0, 0, 0);
		cv::putText(profile_pics[i], "ID" + std::to_string(user_ids[i]), cv::Point(8, 12), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1, 8);
	}

	if (profile_pics.size() > 0)
	{
		cv::Mat combined = imgproc::ImageProc::createOne(profile_pics, 5, 10);
		cv::imshow("Profile Pictures", combined);
		cv::waitKey(0);
		return true;
	}
	else
	{
		std::cout << "No profile pictures taken yet...\n";
	}
	return false;
}


int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);


	cv::Mat color_image;
	io::ImageHandler ih;
	int response_code = 0;

	// config to server connection
	io::TCPClient server_conn;
	server_conn.Config("127.0.0.1", FLAGS_port);
	server_conn.Connect();


	get_profile_pictures(&server_conn);
	

	server_conn.Close();

	return 0;
}
