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
}

void NetworkRequestHandler::addRequest(io::NetworkRequest* request)
{
	mRequestsLock.lock();
	// push back
	mRequests.push(request);
	mRequestsLock.unlock();
}

void NetworkRequestHandler::processRequests()
{
	while (mStatus == RequestHandlerStatus_Running)
	{
		if (!mRequests.empty())
		{
			
#ifdef _DEBUG
			std::cout << "Processing request of type ID(" << mRequests.front()->cRequestID << ") | total: " << mRequests.size() << std::endl;
#endif
			// submit the request to the server
			mRequestsLock.lock();
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

			std::cout << "response id: " << response_identifier << std::endl;
			std::cout << "response id casted: " << (io::NetworkResponseType)response_identifier << std::endl;


			// allocate response
			NetworkResponse* response_ptr = nullptr;
			std::type_index response_type_id = ResponseFactory::AllocateAndLoad((io::NetworkResponseType)response_identifier, socket, response_ptr);

			// add linking
			//mRequestToResponse[request_ptr] = response_ptr;
			//mResponseToRequest[response_ptr] = request_ptr;

			// move to processed stack - sort by response type identifier
			mRespondsLock.lock();
			mResponds[response_type_id].push(response_ptr);
			mRespondsLock.unlock();
		}
	}
}
