#include <thread>
#include <iostream>
#include <queue>
#include <Windows.h>

#include <mutex>
#include <Kinect.Face.h>
#include <opencv2/core.hpp>
#include "io/KinectInterface.h"
#include "tracking/SkeletonTracker.h"

#include "gflags/gflags.h"

enum RequestHandlerStatus
{
	Status_Shutdown = 0,
	Status_Pause = 1,
	Status_Running = 2
};


class Request
{
public:

	Request(int request_id, std::vector<cv::Mat> payload)
	{
		

	}

private:


	std::vector<cv::Mat> mRequest;

};

class RequestHandler
{
public:
	RequestHandler()
	{
		
	}
	~RequestHandler()
	{
		// stop thread
		stop();
	}

	void processRequests()
	{
		while(mStatus == Status_Running)
		{

			if(!mRequests.empty())
			{
				std::cout << "Processing request: " << mRequests.front() << std::endl;
				mRequestsLock.lock();

				// send request id

				// send message length


				mRequests.pop();	// pop front
									// simulate waiting time
				mRequestsLock.unlock();
			}

			Sleep(1000);
		}
	}
	void start()
	{
		mStatus = Status_Running;
		mThread = std::thread(&RequestHandler::processRequests, this);
	}
	void stop()
	{
		mStatus = Status_Shutdown;
		mThread.join();
	}


	// identification request
	

	// training request


	void addRequest(int value)
	{
		mRequestsLock.lock();
		// push back
		mRequests.push(value);

		mRequestsLock.unlock();
	}



private:
	std::thread mThread;
	// tasks to process
	std::queue<int> mRequests;
	std::queue<int> mProcessedRequests;

	int mStatus;

	std::mutex mRequestsLock;
	std::mutex mProcessedRequestsLock;

};

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	RequestHandler handle;

	// start handling requests
	handle.start();
	std::cout << "Starting request handler..." << std::endl;
	Sleep(1000);
	std::cout << "Generating requests..." << std::endl;


	io::KinectSensorMultiSource k;

	cv::Mat color_image;

	// initialize sensor
	if (FAILED(k.Open())) {
		std::cout << "Initialization failed" << std::endl;
		return -1;
	}


	// add 3 requests
	for(int i=0;i<3;i++)
	{
		handle.addRequest(i);
	}
	

	while(1)
	{
		Sleep(1000);
	}
	// blocking call - wait till request handler is finished (processRequests terminates)
	handle.stop();
}