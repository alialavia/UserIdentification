#include "io/RequestHandler.h"
#include <iostream>
// tcp networking
#include <io/Networking.h>
#include "io/RequestTypes.h"
#include "io/ResponseTypes.h"


#ifdef _DEBUG
#include <stdexcept>
#endif


using namespace io;


NetworkRequestHandler::NetworkRequestHandler()
{

}

NetworkRequestHandler::~NetworkRequestHandler()
{

#ifdef _DEBUG_REQUESTHANDLER
	std::cout << "--- Destructing Request Handler " << std::endl;
#endif

	// stop the processing thread thread
	if (mStatus == RequestHandlerStatus_Running) {
		stop();
	}

	// delete all pending requests
	while (!mRequests.empty())
	{
		NetworkRequest* req = mRequests.front();
		delete(req);
		mRequests.pop();
	}
	while (!mPriorityRequests.empty())
	{
		NetworkRequest* req = mPriorityRequests.front();
		delete(req);
		mPriorityRequests.pop();
	}

	// delete all unread responses
	for (auto it = mResponds.begin(); it != mResponds.end(); ++it)
	{
		while (!it->second.empty())
		{
			NetworkResponse* resp = it->second.front();
			delete(resp);
		}
	}
	for (auto it = mPriorityResponds.begin(); it != mPriorityResponds.end(); ++it)
	{
		while (!it->second.empty())
		{
			NetworkResponse* resp = it->second.front();
			delete(resp);
		}
	}
}

void NetworkRequestHandler::addRequest(io::NetworkRequest* request, bool priority)
{
	if(priority)
	{
		mRequestsLock.lock();
		mPriorityRequests.push(request);
		mRequestsLock.unlock();
	}else
	{
		// block request adding till request queue is empty again
		if (GetRequestCount() == mMaxRequests) {
#ifdef _DEBUG_REQUESTHANDLER
			std::cout << "Reached max. request count. Waiting to synchronize with request handler..." << std::endl;
#endif

			while (GetRequestCount() > mContinuationThresh) {
				// wait
				std::cout << "-- refresh" << std::endl;
				Sleep(mRefreshRate);
			}
		}

		mRequestsLock.lock();
		mRequests.push(request);	// push back
		mRequestsLock.unlock();
	}

}

void NetworkRequestHandler::cancelPendingRequest(io::NetworkRequest* request) {
	mRequestsLock.lock();

	// check if request really is pending (may already be processed)
	if(mRequests.contains(request))
	{
		// remove from task queue
		mRequests.erase(request);
		// delete request itself
		delete(request);
	}
	if (mPriorityRequests.contains(request))
	{
		// remove from task queue
		mPriorityRequests.erase(request);
		// delete request itself
		delete(request);
	}
	mRequestsLock.unlock();
}

void NetworkRequestHandler::processAllPendingRequests() {
	math::SequentialContainer<NetworkRequest*>* queue_ptr = nullptr;

	while (!mPriorityRequests.empty() || !mRequests.empty())
	{
		// process priority requests first
		if (!mPriorityRequests.empty())
		{
			queue_ptr = &mPriorityRequests;
		}
		else
		{
			queue_ptr = &mRequests;
		}

		if (!queue_ptr->empty())
		{
			// submit the request to the server
			mRequestsLock.lock();
#ifdef _DEBUG_REQUESTHANDLER
			std::cout << "---------------PROCESSING REQUEST----------------" << std::endl;
			std::cout << "Type ID(" << queue_ptr->front()->cRequestType << ") | total of this type: " << queue_ptr->size() << std::endl;
			std::cout << "-------------------------------------------------" << std::endl;
#endif
			NetworkRequest* request_ptr = queue_ptr->front();
			queue_ptr->pop();	// pop front
			mRequestsLock.unlock();

			// extract server connection
			io::TCPClient* socket = request_ptr->GetServerConnection();

#ifndef _KEEP_SERVER_CONNECTION
			// connect to server
			socket->Connect();
#endif

			// submit
			request_ptr->SubmitRequest();

			// wait for response from this socket - blocking
			// response factory
			int response_identifier = socket->Receive32bit<int>();

			NetworkResponse* response_ptr = nullptr;
			// allocate response
			std::type_index response_type_id = ResponseFactory::AllocateAndLoad((io::NetworkResponseType)response_identifier, socket, response_ptr);
			// add linking
			mMappingLock.lock();
			mResponseToRequest[response_ptr] = request_ptr;
			mRequestToResponse[request_ptr] = response_ptr;
			mMappingLock.unlock();

			mRespondsLock.lock();
			if (queue_ptr == &mPriorityRequests)
			{
				mPriorityResponds[response_type_id].push(response_ptr);
			}
			else
			{
				mResponds[response_type_id].push(response_ptr);
			}
			mRespondsLock.unlock();

#ifndef _KEEP_SERVER_CONNECTION
			// disconnect from server
			socket->Close();

#ifdef _DEBUG_REQUESTHANDLER
			std::cout << "--- Request terminated - Disconnected from server " << std::endl;
#endif

#endif
		}
	}

}

void NetworkRequestHandler::processRequests()
{

#ifdef _DEBUG_REQUESTHANDLER
	std::cout << "--- Starting NetworkRequestHandler::processRequests()" << std::endl;
#endif
	math::SequentialContainer<NetworkRequest*>* queue_ptr = nullptr;

	while (mStatus == RequestHandlerStatus_Running)
	{
		// process priority requests first
		if(!mPriorityRequests.empty())
		{
			queue_ptr = &mPriorityRequests;
		}else
		{
			queue_ptr = &mRequests;
		}

		if (!queue_ptr->empty())
		{
			// submit the request to the server
			mRequestsLock.lock();
#ifdef _DEBUG_REQUESTHANDLER
			std::cout << "---------------PROCESSING REQUEST----------------" << std::endl;
			std::cout << "Type ID(" << queue_ptr->front()->cRequestType << ") | total of this type: " << queue_ptr->size() << std::endl;
			std::cout << "-------------------------------------------------" << std::endl;
#endif
			NetworkRequest* request_ptr = queue_ptr->front();
			queue_ptr->pop();	// pop front
			mRequestsLock.unlock();

			// extract server connection
			io::TCPClient* socket = request_ptr->GetServerConnection();

#ifndef _KEEP_SERVER_CONNECTION
			// connect to server
			socket->Connect();
#endif

			// submit
			request_ptr->SubmitRequest();

			// wait for response from this socket - blocking
			// response factory
			int response_identifier = socket->Receive32bit<int>();

#ifdef _DEBUG_REQUESTHANDLER
			//std::cout << "--- Response id: " << response_identifier << " | Waiting for response data" << std::endl;
#endif

			NetworkResponse* response_ptr = nullptr;
			// allocate response
			std::type_index response_type_id = ResponseFactory::AllocateAndLoad((io::NetworkResponseType)response_identifier, socket, response_ptr);

			// move to processed stack - sort by response type identifier
			
			// add linking
			mMappingLock.lock();
			mResponseToRequest[response_ptr] = request_ptr;
			mRequestToResponse[request_ptr] = response_ptr;
			mMappingLock.unlock();

			mRespondsLock.lock();
			if(queue_ptr == &mPriorityRequests)
			{
				mPriorityResponds[response_type_id].push(response_ptr);
			}else
			{
				mResponds[response_type_id].push(response_ptr);
			}
			mRespondsLock.unlock();

#ifndef _KEEP_SERVER_CONNECTION
			// disconnect from server
			socket->Close();

#ifdef _DEBUG_REQUESTHANDLER
			std::cout << "--- Request terminated - Disconnected from server " << std::endl;
#endif

#endif

		}
	}

#ifdef _DEBUG_REQUESTHANDLER
	std::cout << "--- NetworkRequestHandler processRequests() terminated" << std::endl;
#endif

}
