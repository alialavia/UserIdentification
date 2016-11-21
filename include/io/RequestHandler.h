#ifndef IO_REQUESTHANDLER_H_
#define IO_REQUESTHANDLER_H_

#include <queue>
#include <map>
#include <mutex>
#include <thread>

#include <opencv2/core.hpp>
#include "ResponseTypes.h"

namespace io {

	class NetworkRequest;

	// forward declarations
	class TCPClient;

	enum RequestHandlerStatus
	{
		RequestHandlerStatus_Shutdown = 0,
		RequestHandlerStatus_Pause = 1,
		RequestHandlerStatus_Running = 2
	};

	class NetworkRequestHandler
	{
	public:
		NetworkRequestHandler();
		~NetworkRequestHandler();

		void start()
		{
			mStatus = RequestHandlerStatus_Running;
			mThread = std::thread(&NetworkRequestHandler::processRequests, this);
		}
		void stop()
		{
			mStatus = RequestHandlerStatus_Shutdown;
			mThread.join();
		}

		// ok
		void addRequest(io::NetworkRequest* request);

		// ok
		template<class T>
		bool PopResponse(T *response_container)
		{
			bool status = false;
			mRespondsLock.lock();

			if (
				mResponds.count(typeid(T)) > 0 &&
				mResponds[typeid(T)].size() > 0
				) {

				// load response from specific request type
				NetworkResponse* response = mResponds[typeid(T)].front();

				// remove from queue - pop front
				mResponds[typeid(T)].pop();

				// copy response to response container
				memcpy(response_container, response, sizeof(T));

				mMappingLock.lock();
				// delete corresponding request and mapping
				std::map<NetworkResponse*, NetworkRequest*>::iterator it1 = mResponseToRequest.find(response);
				delete(it1->second);	// delete request
				mResponseToRequest.erase(it1);	// delete map item
				mMappingLock.unlock();

				// delete original response container
				delete(response);
			}

			mRespondsLock.unlock();
			return status;
		}

		// ok
		void processRequests();


	private:
		int mStatus;			// status of the request handler
		std::thread mThread;	// processing thread

		// unprocessed requests in submit order
		std::queue<NetworkRequest*> mRequests;
		// processed requests ordered by request type in processing order (stacked)
		std::map<std::type_index, std::queue<NetworkResponse*>> mResponds;

		// linking
		std::map<NetworkResponse*, NetworkRequest*> mResponseToRequest;

		std::mutex mMappingLock;
		std::mutex mRequestsLock;
		std::mutex mRespondsLock;
	};

};

#endif
