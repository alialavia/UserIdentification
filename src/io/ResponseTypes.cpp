#include "io/ResponseTypes.h"
#include <stdexcept>


using namespace io;

std::type_index ResponseFactory::AllocateAndLoad(NetworkResponseType type_id, io::TCPClient* conn, NetworkResponse* &ptr)
{

	void* response = nullptr;

	// TODO: cleanup/auto type detection

	if (type_id == NetworkResponse_Error)
	{
		ErrorResponse* resp = nullptr;
		resp = new ErrorResponse(conn);
		resp->GetPayload();
		ptr = resp;
		return typeid(ErrorResponse);

	}
	else if(type_id == NetworkResponse_OK)
	{
		OKResponse* resp = nullptr;
		resp = new OKResponse(conn);
		resp->GetPayload();
		ptr = resp;
		return typeid(OKResponse);

	}else if (type_id == NetworkResponse_Identification)
	{
		IdentificationResponse* resp = nullptr;
		resp = new IdentificationResponse(conn);
		resp->GetPayload();
		ptr = resp;
		return typeid(IdentificationResponse);

	}else if(type_id == NetworkResponse_Embedding){
		EmbeddingResponse* resp = nullptr;
		resp = new EmbeddingResponse(conn);
		resp->GetPayload();
		ptr = resp;
		return typeid(EmbeddingResponse);
	}else
	{
		ErrorResponse* resp = nullptr;
		resp = new ErrorResponse(conn);
		resp->mMessage = "Invalid response type id";
		ptr = resp;
		return typeid(ErrorResponse);
	}

}
