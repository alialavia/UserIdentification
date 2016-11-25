#include "io/RequestTypes.h"
#include <ostream>
#include <iostream>
#include <io/Networking.h>


using namespace io;

// ------------ NETWORKING

NetworkRequest::NetworkRequest(io::TCPClient* server_conn, NetworkRequestType req_id) : cRequestType(req_id), pServerConn(server_conn){
#ifdef _DEBUG
	if (server_conn == nullptr) {
		std::cout << "Network request failed: Invalid server connection" << std::endl;
	}
#endif
}

bool NetworkRequest::SubmitRequest()
{

	// send request id to server as uchar (0-255)
	int bytecount;
	bytecount = pServerConn->SendUChar(cRequestType);

	// send payload
	SubmitPayload();

	return true;
}
