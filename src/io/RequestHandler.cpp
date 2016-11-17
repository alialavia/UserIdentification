#include "io/RequestHandler.h"
#include <iostream>
// tcp networking
#include <io\Networking.h>


#ifdef _DEBUG
#include <stdexcept>
#endif


using namespace io;


// -------- Base Request Type



// -------- Network Requests

NetworkRequest::NetworkRequest(io::TCPClient* server_conn) : pServerConn(server_conn) {
#ifdef _DEBUG
	if (server_conn == nullptr) {
		std::cout << "Network request failed: Invalid server connection" << std::endl;
	}
#endif
}

void NetworkRequest::SubmitRequestID() {
	// send request id to server as uchar (0-255)
	pServerConn->SendUChar(mRequestID);
}

bool NetworkRequest::SubmitRequest()
{
	// request terminated - reconnect to server
	if (!pServerConn->Reconnect())
	{
		std::cout << "Could not connect to server..." << std::endl;
		return false;
	}

	// send request id
	SubmitRequestID();

	// send payload
	SubmitPayload();

	return true;
}

int NetworkRequest::GetRequestID() {
	return mRequestID;
}

// -- Single Image Identification Request
void IdentificationRequestSingleImage::SubmitPayload() {
	
#ifdef _DEBUG
	if (mImage.size().width != mImage.size().height) {
		throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
	}
#endif
	// send image dimension
	pServerConn->SendUInt(mImage.size().width);

	// send image
	pServerConn->SendRGBImage(mImage);
}

void IdentificationRequestSingleImage::AcquireResponse(){
	// get user id
	mUserID = pServerConn->Receive32bit<int>();
	// get probability
	mProbability = pServerConn->Receive32bit<float>();
}

void IdentificationRequestSingleImage::GetResponse(IdentificationResponse& response) {
	response.mProbability = mProbability;
	response.mUserID = mUserID;
}


// -------- Threaded Request Handler

NetworkRequestHandler::NetworkRequestHandler()
{

}
NetworkRequestHandler::~NetworkRequestHandler()
{
	// stop the processing thread thread
	stop();
}

void NetworkRequestHandler::processRequests()
{
	while (mStatus == Status_Running)
	{
		if (!mRequests.empty())
		{
			
#ifdef _DEBUG
			std::cout << "Processing request of type ID(" << mRequests.front()->GetRequestID() << ") | total: " << mRequests.size() << std::endl;
#endif
			// submit the request to the server
			mRequestsLock.lock();
			NetworkRequest* oldest_req = mRequests.front();
			mRequests.pop();	// pop front
			// submit
			oldest_req->SubmitRequest();
			mRequestsLock.unlock();

			// wait for response - blocking
			oldest_req->AcquireResponse();

			// move to processed stack
			mProcessedRequestsLock.lock();
			mProcessedRequests[oldest_req->GetRequestID()].push(oldest_req);
			mProcessedRequestsLock.unlock();
		}
	}
}

void NetworkRequestHandler::addRequest(io::NetworkRequest* request)
{
	mRequestsLock.lock();
	// push back
	mRequests.push(request);
	mRequestsLock.unlock();
}

template<typename T>
bool NetworkRequestHandler::PopResponse(int request_type, T &response_container) {

	bool status = false;
	mProcessedRequestsLock.lock();

	if (
		mProcessedRequests.count(request_type) > 0 && 
		mProcessedRequests[request_type].size() > 0
		) {


		// load response from specific request type
		NetworkRequest* processed_req = mProcessedRequests[request_type].front();

		// remove from queue - pop front
		mProcessedRequests[request_type].pop();

		// copy response to response container
		processed_req->GetResponse(response_container);

		// delete request container
		delete(processed_req);
	}

	mProcessedRequestsLock.unlock();
	return status;
}