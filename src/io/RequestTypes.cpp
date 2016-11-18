#include "io/RequestTypes.h"
#include <ostream>
#include <iostream>
#include <io/Networking.h>


using namespace io;

Request::Request() :pR(nullptr)
{

}

Request::~Request()
{
	// cleanup response
	if (pR) {
		delete(pR);
		pR = nullptr;
	}
}

void NetworkRequest::AcquireResponse()
{
	// get response identifier
	int response_identifier = pServerConn->Receive32bit<int>();

	// allocate response container
	pR = ResponseFactory::AllocateAndLoad((NetworkResponseType)response_identifier, pServerConn);

	// get response type

}


// ------------ NETWORKING

NetworkRequest::NetworkRequest(io::TCPClient* server_conn, NetworkRequestType req_id) : cRequestID(req_id), pServerConn(server_conn), pResponse(nullptr) {
#ifdef _DEBUG
	if (server_conn == nullptr) {
		std::cout << "Network request failed: Invalid server connection" << std::endl;
	}
#endif
}

bool NetworkRequest::SubmitRequest()
{
	// request terminated - reconnect to server
	if (!pServerConn->Reconnect())
	{
		std::cout << "Could not connect to server..." << std::endl;
		return false;
	}

	// send request id to server as uchar (0-255)
	pServerConn->SendUChar(cRequestID);

	// send payload
	SubmitPayload();

	return true;
}

