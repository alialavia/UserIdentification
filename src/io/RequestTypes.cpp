#include "io/RequestTypes.h"
#include <ostream>
#include <iostream>
#include <io/Networking.h>


using namespace io;

// ------------ NETWORKING

NetworkRequest::NetworkRequest(io::TCPClient* server_conn, NetworkRequestType req_id) : cRequestID(req_id), pServerConn(server_conn){
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
