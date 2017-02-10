#ifndef IO_RESPONSETYPES_H_
#define IO_RESPONSETYPES_H_

#include <typeindex>
#include <string>
#include <opencv2\core.hpp>
#include <io/Networking.h>

namespace io {
	class NetworkResponse;
	class TCPClient;

	// response IDs: received from server
	enum NetworkResponseType
	{
		NetworkResponse_Identification = 1,
		NetworkResponse_Embedding = 2,
		NetworkResponse_Image = 3,
		NetworkResponse_ImageQuadratic = 4,
		NetworkResponse_Reidentification = 10,
		NetworkResponse_Error = 999,
		NetworkResponse_OK = 111,
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
		virtual void GetPayload() = 0;
	};

	// networking
	class NetworkResponse: public Response
	{
	public:
		NetworkResponse(io::TCPClient* conn, NetworkResponseType type): pConn(conn), cTypeID(type){}
		bool Load(int* response_type = nullptr) {
			int identifier = pConn->Receive32bit<int>();
			bool succ = (identifier == cTypeID);
			if (succ) {
				GetPayload();	// load response data
			}else
			{
				*response_type = identifier;
			}
			return succ;
		}
		io::TCPClient* pConn;
		NetworkResponseType cTypeID;
	};


	// -----------------------------------------
	//		ADD CUSTOM IMPLEMENTATIONS HERE
	// -----------------------------------------
	// - add NetworkResponseType
	// - add Generation in Factory

	// specific response types
	class ErrorResponse : public NetworkResponse
	{
	public:
		ErrorResponse(io::TCPClient* conn = nullptr) :NetworkResponse(conn, NetworkResponse_Error){}
		void GetPayload() { mMessage = pConn->ReceiveStringWithVarLength(); }
		std::string mMessage = "Invalid response";

	};

	class OKResponse : public NetworkResponse
	{
	public:
		OKResponse(io::TCPClient* conn = nullptr) :NetworkResponse(conn, NetworkResponse_OK){}
		void GetPayload() { mMessage = pConn->ReceiveStringWithVarLength(); };
		std::string mMessage = "Request processed successfully";
	};

	class IdentificationResponse : public NetworkResponse
	{
	public:
		IdentificationResponse(io::TCPClient* conn = nullptr):NetworkResponse(conn, NetworkResponse_Identification)
		{
		}
		IdentificationResponse(const IdentificationResponse& other) :NetworkResponse(other)
		{
			mImage = other.mImage.clone();	// make deep copy
			mUserID = other.mUserID;
			mUserNiceName = other.mUserNiceName;
			mConfidence = other.mConfidence;
		}
		void GetPayload() {
			mUserID = pConn->Receive32bit<int>();
			mUserNiceName = pConn->ReceiveStringWithVarLength();
			mConfidence = (int)pConn->Receive8bit<uint8_t>();
			bool hasProfilePicture = pConn->Receive8bit<bool>();
			if(hasProfilePicture)
			{
				pConn->ReceiveRGBImageQuadratic(mImage);
			}
		};
		int mUserID = -1;
		std::string mUserNiceName = "";
		cv::Mat mImage;	// profile picture
		int mConfidence = 0;	// current confidence value in [%]
	};

	class EmbeddingResponse : public NetworkResponse
	{
	public:
		EmbeddingResponse(io::TCPClient* conn = nullptr):NetworkResponse(conn, NetworkResponse_Embedding)
		{
		}
		void GetPayload() {
			// load 128 dimensional vector
			for (size_t i = 0; i<cNrEmbeddings; i++)
			{
				mEmbedding[i] = pConn->Receive64bit<double>();
			}
		}; // receive response specific data from server
		static const int cNrEmbeddings = 128;
		double mEmbedding[cNrEmbeddings];
	};

	class ImageResponse : public NetworkResponse
	{
	public:
		ImageResponse(io::TCPClient* conn = nullptr) :NetworkResponse(conn, NetworkResponse_Image){}
		void GetPayload() {pConn->ReceiveRGBImage(mImage);};
		cv::Mat mImage;
	};

	class QuadraticImageResponse : public NetworkResponse
	{
	public:
		QuadraticImageResponse(io::TCPClient* conn = nullptr) :NetworkResponse(conn, NetworkResponse_ImageQuadratic) {}
		QuadraticImageResponse(const QuadraticImageResponse& other) :NetworkResponse(other) { mImage = other.mImage.clone(); }
		void GetPayload() { pConn->ReceiveRGBImageQuadratic(mImage); };
		cv::Mat mImage;
	};

	class ReidentificationResponse : public NetworkResponse
	{
	public:
		ReidentificationResponse(io::TCPClient* conn = nullptr) :NetworkResponse(conn, NetworkResponse_Reidentification) {}
		void GetPayload() {};
	};

}

#endif