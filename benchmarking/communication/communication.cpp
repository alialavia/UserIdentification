#include <iostream>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include <io/RequestTypes.h>
#include <gflags/gflags.h>
#include "io/CSVHandling.h"
#include <chrono>

typedef io::Pong Resp;
typedef io::ImageIdentification Req;

DEFINE_string(log_name, "", "Output log (e.g. image_send_log.csv)");
DEFINE_int32(port, 8080, "Server port");
DEFINE_int32(nr_requests, 5, "Number of requests");
DEFINE_int32(nr_images, 10, "Number of images per request");
DEFINE_int32(img_size, 96, "Image dimension (squared image)");

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// config to server connection
	io::TCPClient server_conn;
	server_conn.Config("127.0.0.1", FLAGS_port);

	// params
	int nr_requests = FLAGS_nr_requests;

	std::vector<int> timings_ms;
	// dummy image
	cv::Mat black_img(FLAGS_img_size, FLAGS_img_size, CV_8UC3, cv::Scalar(0, 0, 0));
	std::vector<cv::Mat> dummy_images(FLAGS_nr_images, black_img);
	Resp response(&server_conn);
	Req request(&server_conn, dummy_images);

	// perform requests
	for (int i = 0; i < nr_requests; i++) {
		server_conn.Connect();
		auto t1 = std::chrono::high_resolution_clock::now();

		request.SubmitRequest();

		// wait for reponse
		if (!response.Load()) {
			std::cout << "--- An error occurred during request: ResponseType " << std::endl;
		}
		else {

		}
		auto t2 = std::chrono::high_resolution_clock::now();
		server_conn.Close();

		timings_ms.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
	}

	std::string log_name;
	if(FLAGS_log_name != "")
	{
		log_name = FLAGS_log_name;
	}else
	{
		log_name = std::to_string(FLAGS_img_size) + "x" + std::to_string(FLAGS_img_size) + " - " + std::to_string(FLAGS_nr_images) + " images per batch.csv";
	}
	 
	std::cout << "--- saving log to: " + log_name << std::endl;

	io::CSVWriter csv(log_name);
	csv.addList(timings_ms);

	return 0;
}
