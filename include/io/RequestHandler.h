#ifndef IO_REQUESTHANDLER_H_
#define IO_REQUESTHANDLER_H_

#include <queue>
#include <map>
#include <mutex>
#include <thread>
#include <math\Math.h>

#include <opencv2/core.hpp>
#include "ResponseTypes.h"

#define _DEBUG_REQUESTHANDLER

namespace io {

	// forward declarations
	class TCPClient;
	class NetworkRequest;
	enum NetworkRequestType;

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

		void cancelPendingRequest(io::NetworkRequest* request);

		template<class T>
		bool PopResponse(T *response_container, NetworkRequest* &req_lookup, NetworkRequestType* request_type = nullptr)
		{
			bool status = false;
			mRespondsLock.lock();

			if (
				mResponds.count(typeid(T)) > 0 &&
				mResponds[typeid(T)].size() > 0
				) {

#ifdef _DEBUG_REQUESTHANDLER
				std::cout << "--- PopResponse: " << typeid(T).name() << " | response count: " << mResponds[typeid(T)].size() << std::endl;
#endif

				// load response from specific request type
				NetworkResponse* response = mResponds[typeid(T)].front();

				// remove from queue - pop front
				mResponds[typeid(T)].pop();

				// copy response to specific response container
				memcpy(response_container, response, sizeof(T));

				mMappingLock.lock();
				NetworkRequest* request = mResponseToRequest[response];

				// provide req lookup key
				req_lookup = request;
				if(request_type != nullptr)
				{
					*request_type = request->cRequestType;	// share request type
				}
				
				// delete req>resp linking
				mResponseToRequest.erase(response);
				mRequestToResponse.erase(request);
				mMappingLock.unlock();

				// delete response
				delete(response);
				// delete request
				delete(request);

				status = true;

				// ---------------------------

				//if (mResponseToRequest.count(response) == 0) {
				//	std::cout << "We got a problem here! Response > Request mapping is corrupted." << std::endl;
				//	throw std::invalid_argument("We got a problem here! Response > Request mapping is corrupted.");
				//}

				//req_lookup = mResponseToRequest[response];

				//// check if request is not deleted yet
				//if (req_lookup != nullptr) {
				//	if (request_type != nullptr) {
				//		*request_type = req_lookup->cRequestType;	// share request type
				//	}
				//	// ------ remove request
				//	delete(req_lookup);
				//	mRequestToResponse.erase(req_lookup);
				//}
				//else {
				//	std::cout << "--- Request already deleted!" << std::endl;
				//}

				//// delete mapping
				//mResponseToRequest.erase(response);

				//// open lock
				//mMappingLock.unlock();

				//// delete original response container
				//delete(response);
				//status = true;

				// --------------------------------------------------

				
				//// delete corresponding request and mapping
				//std::map<NetworkResponse*, NetworkRequest*>::iterator it1 = mResponseToRequest.find(response);

				//if (it1 == mResponseToRequest.end()) {
				//	std::cout << "We got a problem here!" << std::endl;
				//	throw std::invalid_argument("We got a problem here!");
				//}

				//// TODO: fix situation, when request is already deleted (User has left the scene)

				//// ------ returns
				//req_lookup = it1->second; // share request pointer

				//// check if request is not deleted yet
				//if (req_lookup != nullptr) {

				//	std::cout << "--- request pointer is not zero but!!!" << std::endl;

				//	if (request_type != nullptr) {
				//		*request_type = it1->second->cRequestType;	// share request type
				//	}
				//	// ------ cleanup
				//	delete(req_lookup);
				//}
				//else {
				//	std::cout << "--- Request already deleted!" << std::endl;
				//}


				//if (mResponseToRequest.count(response)==0) {
				//	std::cout << "--- Key (response pointers) is not in map!" << std::endl;
				//}
				//
				//// delete request
				//mResponseToRequest.erase(it1);	// delete map item
				//mMappingLock.unlock();

				//// delete original response container
				//delete(response);
				//status = true;
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
		math::SequentialContainer<NetworkRequest*> mRequests;
		// processed requests ordered by request type in processing order (stacked)
		std::map<std::type_index, std::queue<NetworkResponse*>> mResponds;

		// linking
		// TODO: use smart pointers
		std::map<NetworkResponse*, NetworkRequest*> mResponseToRequest;
		std::map<NetworkRequest*, NetworkResponse*> mRequestToResponse;	// helps during cleanup

		// TODO: use single lock (processRequests might cause problems otherwise)
		std::mutex mMappingLock;
		std::mutex mRequestsLock;
		std::mutex mRespondsLock;
	};

};

#endif
