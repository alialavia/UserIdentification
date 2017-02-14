#ifndef IO_REQUESTHANDLER_H_
#define IO_REQUESTHANDLER_H_

#include <queue>
#include <map>
#include <mutex>
#include <thread>
#include <math\Math.h>

#include <opencv2/core.hpp>
#include <io/ResponseTypes.h>
#include <io/RequestTypes.h>

#define _DEBUG_REQUESTHANDLER

namespace io {

	typedef io::QuadraticImageResponse QIR;

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

		size_t GetRequestCount(){
			size_t count = 0;
			mRequestsLock.lock();
			count = mRequests.size();
			mRequestsLock.unlock();
			return count;
		}

		template<class T>
		size_t GetResponseCount() {
			size_t count = 0;
			mRespondsLock.lock();
			count = (mResponds.count(typeid(T)) > 0 ? mResponds[typeid(T)].size() : 0);
			mRespondsLock.unlock();
			return count;
		}

		void addRequest(io::NetworkRequest* request, bool priority=false);

		void cancelPendingRequest(io::NetworkRequest* request);


		template<class T>
		bool PopResponse(T *response_container, NetworkRequest* &req_lookup, NetworkRequestType* request_type = nullptr)
		{
			bool status = false;
			std::map<std::type_index, std::queue<NetworkResponse*>> *queue_ptr = nullptr;
			mRespondsLock.lock();

			if (
				mPriorityResponds.count(typeid(T)) > 0 &&
				mPriorityResponds[typeid(T)].size() > 0
				) {
				queue_ptr = &mPriorityResponds;
			}else
			{
				queue_ptr = &mResponds;
			}


			if (
				queue_ptr->count(typeid(T)) > 0 &&
				(*queue_ptr)[typeid(T)].size() > 0
				) {

#ifdef _DEBUG_REQUESTHANDLER
				std::cout << "--- PopResponse: " << typeid(T).name() << " | response count: " << (*queue_ptr)[typeid(T)].size() << std::endl;
#endif

				// load response from specific request type
				NetworkResponse* response = (*queue_ptr)[typeid(T)].front();

				// remove from queue - pop front
				(*queue_ptr)[typeid(T)].pop();

				// variant 1 ---------------- type specific
				//copyImageContainer(std::integral_constant<bool, is_same<QIR, T>::value>(), response, response_container);

				// variant 2 ---------------- custom assignement operator
				// 1. cast to container class
				T* spec_resp = dynamic_cast<T*>(response);
				// 2. copy by copy constructor
				*response_container = T(*spec_resp);

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
			}

			mRespondsLock.unlock();
			return status;
		}

		void processRequests();

		// ------------------------- deprecated

		template<class T, class U>
		struct is_same { static const bool value = false; };

		template<class T>
		struct is_same<T, T> { static const bool value = true; };

		template<class T>
		void copyImageContainer(std::true_type, NetworkResponse* response, T *response_container)
		{
			// copy individual fields manually
			QIR* qir_pointer = dynamic_cast<QIR*>(response);
			response_container->mImage = qir_pointer->mImage.clone();

		}
		template<class T>
		void copyImageContainer(std::false_type, NetworkResponse* response, T *response_container)
		{
			// copy all data
			memcpy(response_container, response, sizeof(T));
		}

	private:

		const int mMaxRequests = 10;		// maximum queued requests
		const int mContinuationThresh = 5;	// request processing threshold
		const int mRefreshRate = 100;		// refresh rate to check mContinuationThresh

		int mStatus;			// status of the request handler
		std::thread mThread;	// processing thread

		// unprocessed requests in submit order
		math::SequentialContainer<NetworkRequest*> mRequests;
		// priority queue
		math::SequentialContainer<NetworkRequest*> mPriorityRequests;
		// processed requests ordered by request type in processing order (stacked)
		std::map<std::type_index, std::queue<NetworkResponse*>> mResponds;
		// priority responses
		std::map<std::type_index, std::queue<NetworkResponse*>> mPriorityResponds;

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
