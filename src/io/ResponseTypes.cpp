#include "io/ResponseTypes.h"
#include <stdexcept>
#include <io/Networking.h>

using namespace io;

std::type_index ResponseFactory::AllocateAndLoad(NetworkResponseType type_id, io::TCPClient* conn, NetworkResponse* &ptr)
{

	void* response = nullptr;

	// TODO: cleanup/auto type detection
	if (type_id == NetworkResponse_IdentificationResponse)
	{
		IdentificationResponse* resp = nullptr;
		resp = new IdentificationResponse(conn);
		resp->Load();
		ptr = resp;
		return typeid(IdentificationResponse);

	}else if(type_id == NetworkResponse_EmbeddingResponse){
		EmbeddingResponse* resp = nullptr;
		resp = new EmbeddingResponse(conn);
		resp->Load();
		ptr = resp;
		return typeid(EmbeddingResponse);
	}else
	{
		ErrorResponse* resp = nullptr;
		resp = new ErrorResponse(conn);
		resp->mMessage = "Invalid response type id";
		resp->Load();
		ptr = resp;
		return typeid(ErrorResponse);
	}

}

// ------------ RESPONSE DEFINITIONS

void IdentificationResponse::Load()
{
	// load specific data
	mUserID = pConn->Receive32bit<int>();
	mUserNiceName = pConn->ReceiveStringWithVarLength();
	mProbability = pConn->Receive32bit<float>();
}

void EmbeddingResponse::Load()
{
	// load 128 dimensional vector
	for(size_t i=0;i<cNrEmbeddings;i++)
	{
		mEmbedding[i] = pConn->Receive64bit<double>();
	}
}


