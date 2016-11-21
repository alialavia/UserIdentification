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
#ifdef _DEBUG
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

void NetworkRequestHandler::processRequests()
{
	while (mStatus == RequestHandlerStatus_Running)
	{
		if (!mRequests.empty())
		{
			// submit the request to the server
			mRequestsLock.lock();
#ifdef _DEBUG
			std::cout << "--- Processing request of type ID(" << mRequests.front()->cRequestID << ") | total: " << mRequests.size() << std::endl;
#endif
			NetworkRequest* request_ptr = mRequests.front();
			mRequests.pop();	// pop front
			mRequestsLock.unlock();

			// submit
			request_ptr->SubmitRequest();

			// extract server connection
			io::TCPClient* socket = request_ptr->GetServerConnection();

			// wait for response from this socket - blocking
			// response factory
			int response_identifier = socket->Receive32bit<int>();

			NetworkResponse* response_ptr = nullptr;
			// allocate response
			std::type_index response_type_id = ResponseFactory::AllocateAndLoad((io::NetworkResponseType)response_identifier, socket, response_ptr);

			// add linking
			mResponseToRequest[response_ptr] = request_ptr;

			// move to processed stack - sort by response type identifier
			mMappingLock.lock();
			mResponds[response_type_id].push(response_ptr);
			mMappingLock.unlock();


		}
	}
}
