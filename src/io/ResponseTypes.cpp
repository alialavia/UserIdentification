#include "io/ResponseTypes.h"
#include <stdexcept>
#include <io/Networking.h>

using namespace io;

std::type_index ResponseFactory::AllocateAndLoad(NetworkResponseType type_id, io::TCPClient* conn, NetworkResponse* &ptr)
{

	void* response = nullptr;

	if (type_id == NetworkResponse_IdentificationResponse)
	{
		IdentificationResponse* resp = nullptr;
		resp = new IdentificationResponse(conn);
		resp->Load();
		ptr = resp;
		return typeid(IdentificationResponse);

	}else
	{
		ErrorResponse* resp = nullptr;
		resp = new ErrorResponse(conn);
		resp->mMessage = "Invalid response type id";
		resp->Load();
		ptr = resp;
		return typeid(IdentificationResponse);
	}

}

// ------------ RESPONSE DEFINITIONS

void IdentificationResponse::Load()
{
	// load specific data
	mUserID = pConn->Receive32bit<int>();
	mProbability = pConn->Receive32bit<float>();
}