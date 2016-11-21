#ifndef IO_REQUESTTYPES_H_
#define IO_REQUESTTYPES_H_
#include "ResponseTypes.h"
#include <opencv2/core/mat.hpp>
#include <io/Networking.h>

namespace io {
	class TCPClient;


	// request base class
	class Request
	{
	};

	// ------------ NETWORKING REQUEST TYPE

	// define network request types (request ids are parsed on the server side)
	enum NetworkRequestType
	{
		NetworkRequest_SingleImageIdentification = 1,
		NetworkRequest_BatchImageIdentification = 2,
	};

	// request type
	class NetworkRequest : public Request
	{
	public:
		NetworkRequest(io::TCPClient* server_conn, NetworkRequestType req_id);

		// submits the request over the network
		// 1. Submit request id
		// 2: Submit payload (defined in explicite implementation)
		bool SubmitRequest();

		io::TCPClient* GetServerConnection()
		{
			return pServerConn;
		}

		const NetworkRequestType cRequestID;
	protected:
		io::TCPClient* pServerConn;

	protected:
		// submit specific payload (data size and data)
		virtual void SubmitPayload() = 0;

	};

	// -----------------------------------------
	//		ADD CUSTOM IMPLEMENTATIONS HERE
	// -----------------------------------------
	// - add NetworkRequestType
	// - add Generation in Factory

	class IdentificationRequestSingleImage : public NetworkRequest
	{
	public:
		IdentificationRequestSingleImage(
			io::TCPClient* server_conn,
			cv::Mat img
		) :
			NetworkRequest(server_conn, NetworkRequest_SingleImageIdentification),
			mImage(img)
		{
		}

	private:

		// submit specific payload
		void SubmitPayload() {

#ifdef _DEBUG
			if (mImage.size().width != mImage.size().height) {
				throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
			}
#endif
			std::cout << mImage.size().width << std::endl;


			// TODO: fix this - size not received by server
			// send image dimension
			pServerConn->SendUInt(mImage.size().width);

			// send image
			pServerConn->SendRGBImage(mImage);
		}

		// payload: quadratic(!) image
		cv::Mat mImage;

	};


}

#endif