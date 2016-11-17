#ifndef IO_REQUESTHANDLER_H_
#define IO_REQUESTHANDLER_H_

#include <queue>
#include <map>
#include <mutex>
#include <thread>

#include <opencv2/core.hpp>


namespace io {

	// forward declarations
	class TCPClient;

	enum RequestHandlerStatus
	{
		Status_Shutdown = 0,
		Status_Pause = 1,
		Status_Running = 2
	};

	// define network request types (request ids are parsed on the server side)
	enum NetworkRequestType
	{
		NetworkRequest_SingleImageIdentification = 1,
		NetworkRequest_BatchImageIdentification = 2,
	};


	// request base class
	class Request
	{
	public:
		Request()
		{
		}
	};


	// response types
	struct IdentificationResponse {
		int mUserID = -1;
		float mProbability = 0.0f;
	};


	// request type
	class NetworkRequest: public Request
	{
	public:
		NetworkRequest(io::TCPClient* server_conn);

		// submits the request over the network
		bool SubmitRequest();
		// wait for response - request specific
		virtual void AcquireResponse() = 0;

		int GetRequestID();

	protected:
		// submit the request id to the server
		void SubmitRequestID();

		// submit specific payload (data size and data)
		virtual void SubmitPayload() = 0;

		int mRequestID;
		io::TCPClient* pServerConn;
	};

	class IdentificationRequestSingleImage: public NetworkRequest
	{
	public:
		IdentificationRequestSingleImage(
			io::TCPClient* server_conn,
			cv::Mat img
		): 
			NetworkRequest(server_conn),
			mImage(img),
			mUserID(-1),
			mProbability(0.0f)
		{
			// set request id
			mRequestID = NetworkRequest_SingleImageIdentification;
		}

		void AcquireResponse();
		void GetResponse(IdentificationResponse& response);

	private:
		// submit specific payload
		void SubmitPayload();

		// payload: quadratic(!) image
		cv::Mat mImage;

		// response
		int mUserID;
		float mProbability;
	};




	class NetworkRequestHandler
	{
	/*
	// USAGE:
	cv::Mat myImage;
	// chose request type
	typedef io::IdentificationRequestSingleImage req;

	// start request handler
	NetworkRequestHandler handler();
	handler.start();	// parallel processing

	// generate request
	req* new_request = new req(&server_conn, myImage);
	// submit request
	handler.addRequest(new_request);

	// try to get response
	while(){
	
	}

	*/
	public:
		NetworkRequestHandler();
		~NetworkRequestHandler();

		void start()
		{
			mStatus = Status_Running;
			mThread = std::thread(&NetworkRequestHandler::processRequests, this);
		}
		void stop()
		{
			mStatus = Status_Shutdown;
			mThread.join();
		}

		void processRequests();
		void addRequest(io::NetworkRequest* request);


		template<typename T>
		bool PopResponse(int request_type, T &response_container);

	private:
		int mStatus;			// status of the request handler
		std::thread mThread;	// processing thread
		// unprocessed requests in submit order
		std::queue<NetworkRequest*> mRequests;
		// processed requests ordered by request type in processing order (stacked)
		std::map<int, std::queue<NetworkRequest*>> mProcessedRequests;

		std::mutex mRequestsLock;
		std::mutex mProcessedRequestsLock;
	};

};

#endif
