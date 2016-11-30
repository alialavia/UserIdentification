#include <iostream>
#include <strsafe.h>
#include <opencv2\opencv.hpp>
#include <gflags/gflags.h>
#include <string>
#include <io/Networking.h>
#include <io/CSVHandling.h>
#include <io/RequestTypes.h>
#include <io/ResponseTypes.h>


DEFINE_int32(port, 8080, "Server port");
DEFINE_string(filelist, "", "List of the images");
DEFINE_string(output, "embeddings.csv", "Embeddings output");

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// config to server connection
	io::TCPClient server_conn;
	server_conn.Config("127.0.0.1", FLAGS_port);

	if(FLAGS_filelist == "")
	{
		std::cout << "Please enter a valid filename.\n";
		return -1;
	}

	io::CSVParser<std::string> file_handle(FLAGS_filelist);
	io::CSVWriter csv_out(FLAGS_output);

	// load images
	while (file_handle.IterateRows())
	{
		std::string img_name = file_handle.GetVal(0);

		// load image
		cv::Mat image;
		image = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);   // Read the file

		if (!image.data)                              // Check for invalid input
		{
			std::cout << "Could not open or find the image: " << img_name << std::endl;
			continue;
		}


		server_conn.Connect();

		// generate request
		io::EmbeddingCalculationSingleImage req(&server_conn, image);
		req.SubmitRequest();

		//allocate response
		int response_identifier = server_conn.Receive32bit<int>();
		io::NetworkResponse* response_ptr = nullptr;
		std::type_index response_type_id = io::ResponseFactory::AllocateAndLoad((io::NetworkResponseType)response_identifier, &server_conn, response_ptr);


		// get reponse
		if((io::NetworkResponseType)response_identifier == io::NetworkResponse_EmbeddingResponse)
		{
			io::EmbeddingResponse resp;
			// convert
			memcpy(&resp, response_ptr, sizeof(io::EmbeddingResponse));

			// save to file
			csv_out.addEntry(img_name);
			for(size_t i = 0; i<resp.cNrEmbeddings;i++)
			{
				csv_out.addEntry(resp.mEmbedding[i]);
			}
			csv_out.startNewRow();
		}else
		{
			// invalid response/error - drop
			std::cout << "Got invalid network response\n";
		}

		// cleanup
		delete(response_ptr);

	}

	return 0;
}
