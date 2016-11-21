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
		// pass byck type index

		std::cout << resp->mUserID << std::endl;

		return typeid(IdentificationResponse);

	}else
	{
		std::cout << "INVALID!!!" << std::endl;
		throw std::invalid_argument("This response type is not supported.");
	}
}

// ------------ RESPONSE DEFINITIONS

void IdentificationResponse::Load()
{
	// load specific data
	mUserID = pConn->Receive32bit<int>();
	mProbability = pConn->Receive32bit<float>();
}