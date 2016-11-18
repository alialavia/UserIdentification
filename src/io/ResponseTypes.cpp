#include "io/ResponseTypes.h"
#include <stdexcept>
#include <io/Networking.h>

using namespace io;

void* ResponseFactory::AllocateAndLoad(NetworkResponseType type_id, io::TCPClient* conn)
{

	void* response = nullptr;

	if (type_id == NetworkResponse_IdentificationResponse)
	{
		IdentificationResponse* resp = nullptr;
		resp = new IdentificationResponse(conn);
		resp->Load();
		response = resp;
	}else if (type_id == 2)
	{
		// ...

	}else
	{
		throw std::invalid_argument("This response type is not supported.");
	}


	return response;
}

// ------------ RESPONSE DEFINITIONS

void IdentificationResponse::Load()
{
	mUserID = pConn->Receive32bit<int>();
	mProbability = pConn->Receive32bit<float>();
}