#ifndef IO_RESPONSETYPES_H_
#define IO_RESPONSETYPES_H_

namespace io {
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
		/// <summary>
		/// Allocates respond of specific type corresponding to identifier. See NetworkRequestType
		/// </summary>
		/// <param name="type_id">The type identifier.</param>
		/// <returns>void *.</returns>
		static void* AllocateAndLoad(NetworkResponseType type_id, io::TCPClient* conn);
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
	protected:
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
		IdentificationResponse(io::TCPClient* conn):NetworkResponse(conn)
		{
		}
		void Load(); // receive response specific data from server
		int mUserID = -1;
		float mProbability = 0.0f;
	};


}

#endif