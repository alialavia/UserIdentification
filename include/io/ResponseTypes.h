#ifndef IO_RESPONSETYPES_H_
#define IO_RESPONSETYPES_H_

#include <typeindex>
#include <string>

namespace io {
	class NetworkResponse;
	class TCPClient;

	// response IDs: received from server
	enum NetworkResponseType
	{
		NetworkResponse_IdentificationResponse = 1
	};


	/// <summary>
	/// Response Lookup - Generates response types from response ids
	/// </summary>
	class ResponseFactory
	{
	public:		
		static std::type_index AllocateAndLoad(NetworkResponseType type_id, io::TCPClient* conn, NetworkResponse* &ptr);
	};

	// ------------ RESPONSE DEFINITIONS

	// response interface
	class Response {
	public:
		virtual void Load() = 0;
	};

	// networking
	class NetworkResponse: public Response
	{
	public:
		NetworkResponse(io::TCPClient* conn): pConn(conn)
		{
		}
		io::TCPClient* pConn;
	};


	// -----------------------------------------
	//		ADD CUSTOM IMPLEMENTATIONS HERE
	// -----------------------------------------
	// - add NetworkResponseType
	// - add Generation in Factory

	// specific response types
	class IdentificationResponse : public NetworkResponse
	{
	public:
		IdentificationResponse(io::TCPClient* conn = nullptr):NetworkResponse(conn)
		{
		}
		void Load(); // receive response specific data from server
		int mUserID = -1;
		float mProbability = 0.0f;
	};

	class ErrorResponse : public NetworkResponse
	{
	public:
		ErrorResponse(io::TCPClient* conn = nullptr) :NetworkResponse(conn)
		{
		}
		void Load() {};
		std::string mMessage = "Invalid response";
	};

}

#endif