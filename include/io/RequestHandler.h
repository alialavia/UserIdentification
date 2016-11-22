#ifndef IO_REQUESTHANDLER_H_
#define IO_REQUESTHANDLER_H_

#include <queue>
#include <map>
#include <mutex>
#include <thread>

#include <opencv2/core.hpp>
#include "ResponseTypes.h"

#define _DEBUG_REQUESTHANDLER

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

		int GetRequestCount(){
			int count = 0;
			mRequestsLock.lock();
			count = mRequests.size();
			mRequestsLock.unlock();
			return count;
		}

		template<class T>
		int GetResponseCount() {
			int count = 0;
			mRespondsLock.lock();
			count = (mResponds.count(typeid(T)) > 0 ? mResponds[typeid(T)].size() : 0);
			mRespondsLock.unlock();
			return count;
		}

		void addRequest(io::NetworkRequest* request);

		template<class T>
		bool PopResponse(T *response_container, NetworkRequest* &request_ref)
		{
			bool status = false;
			mRespondsLock.lock();

			if (
				mResponds.count(typeid(T)) > 0 &&
				mResponds[typeid(T)].size() > 0
				) {

#ifdef _DEBUG_REQUESTHANDLER
				std::cout << "--- PopResponse: " << mResponds.count(typeid(T)) << " | response count: " << mResponds[typeid(T)].size() << std::endl;
#endif

				// load response from specific request type
				NetworkResponse* response = mResponds[typeid(T)].front();

				// remove from queue - pop front
				mResponds[typeid(T)].pop();

				// copy response to response container
				memcpy(response_container, response, sizeof(T));

				mMappingLock.lock();
				// delete corresponding request and mapping
				std::map<NetworkResponse*, NetworkRequest*>::iterator it1 = mResponseToRequest.find(response);
				request_ref = it1->second;		// share request pointer
				delete(it1->second);			// delete request
				mResponseToRequest.erase(it1);	// delete map item
				mMappingLock.unlock();

				// delete original response container
				delete(response);
				status = true;
			}

			mRespondsLock.unlock();
			return status;
		}

		void processRequests();

	private:

		const int mMaxRequests = 10;		// maximum queued requests
		const int mContinuationThresh = 5;	// request processing threshold
		const int mRefreshRate = 100;		// refresh rate to check mContinuationThresh

		int mStatus;			// status of the request handler
		std::thread mThread;	// processing thread

		// unprocessed requests in submit order
		std::queue<NetworkRequest*> mRequests;
		// processed requests ordered by request type in processing order (stacked)
		std::map<std::type_index, std::queue<NetworkResponse*>> mResponds;

		// linking
		// TODO: use smart pointers
		std::map<NetworkResponse*, NetworkRequest*> mResponseToRequest;

		std::mutex mMappingLock;
		std::mutex mRequestsLock;
		std::mutex mRespondsLock;
	};

};

#endif
