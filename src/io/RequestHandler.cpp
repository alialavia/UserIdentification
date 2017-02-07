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
	stop();

	// delete all pending requests
	while (!mRequests.empty())
	{
		NetworkRequest* req = mRequests.front();
		delete(req);
		mRequests.pop();
	}

	// delete all unread responses
	for (auto it = mResponds.begin(); it != mResponds.end(); ++it)
	{
		while (!it->second.empty())
		{
			NetworkResponse* resp = it->second.front();
			delete(resp);
			mRequests.pop();
		}
	}
}

void NetworkRequestHandler::addRequest(io::NetworkRequest* request)
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
	mRequestsLock.unlock();
}

void NetworkRequestHandler::processRequests()
{

#ifdef _DEBUG_REQUESTHANDLER
	std::cout << "--- Starting NetworkRequestHandler::processRequests()" << std::endl;
#endif

	while (mStatus == RequestHandlerStatus_Running)
	{
		if (!mRequests.empty())
		{
			// submit the request to the server
			mRequestsLock.lock();
#ifdef _DEBUG_REQUESTHANDLER
			std::cout << "---------------PROCESSING REQUEST----------------" << std::endl;
			std::cout << "Type ID(" << mRequests.front()->cRequestType << ") | total of this type: " << mRequests.size() << std::endl;
			std::cout << "-------------------------------------------------" << std::endl;
#endif
			NetworkRequest* request_ptr = mRequests.front();
			mRequests.pop();	// pop front
			mRequestsLock.unlock();

			// extract server connection
			io::TCPClient* socket = request_ptr->GetServerConnection();

			// connect to server
			socket->Connect();

			// submit
			request_ptr->SubmitRequest();

			// wait for response from this socket - blocking
			// response factory
			int response_identifier = socket->Receive32bit<int>();

#ifdef _DEBUG_REQUESTHANDLER
			std::cout << "--- Response id: " << response_identifier << " | Waiting for response data" << std::endl;
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
			mResponds[response_type_id].push(response_ptr);
			mRespondsLock.unlock();

			// disconnect from server
			socket->Close();

#ifdef _DEBUG_REQUESTHANDLER
			std::cout << "--- Request terminated - Disconnected from server " << std::endl;
#endif

		}
	}

#ifdef _DEBUG_REQUESTHANDLER
	std::cout << "--- NetworkRequestHandler processRequests() terminated" << std::endl;
#endif

}
