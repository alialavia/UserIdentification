#include <iostream>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include <io/RequestTypes.h>
#include <gflags/gflags.h>
#include "io/CSVHandling.h"
#include <chrono>

typedef io::Pong Resp;
typedef io::Ping Req;

DEFINE_string(log_name, "network_log.csv", "Output log (e.g. network_log.csv)");
DEFINE_int32(port, 8080, "Server port");
DEFINE_int32(nr_requests, 2, "Number of requests");
DEFINE_bool(reconnect, false, "Reconnect to server after every request");

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// config to server connection
	io::TCPClient server_conn;
	server_conn.Config("127.0.0.1", FLAGS_port);

	// params
	int nr_requests = FLAGS_nr_requests;

	std::vector<int> timings_ms;
	Resp response(&server_conn);
	Req request(&server_conn);

	if (FLAGS_reconnect) {
		for (int i = 0; i < nr_requests; i++) {
			auto t1 = std::chrono::high_resolution_clock::now();
			server_conn.Connect();
			request.SubmitRequest();

			// wait for reponse
			if (!response.Load()) {
				std::cout << "--- An error occurred during identification: ResponseType " << std::endl;
			}
			else {

			}

			server_conn.Close();
			auto t2 = std::chrono::high_resolution_clock::now();
			timings_ms.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
		}
	}
	else {
		server_conn.Connect();
		for (int i = 0; i < nr_requests; i++) {
			auto t1 = std::chrono::high_resolution_clock::now();
			std::cout << "... Sending request...\n";
			request.SubmitRequest();
			// wait for reponse
			if (!response.Load()) {
				std::cout << "--- An error occurred during identification: ResponseType " << std::endl;
			}
			else {

			}
			auto t2 = std::chrono::high_resolution_clock::now();
			timings_ms.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
			// sleep for 1 sec
			Sleep(1000);
		}
		server_conn.Close();
	}

	//io::CSVWriter csv(FLAGS_log_name);
	//csv.addList(timings_ms);

	return 0;
}
