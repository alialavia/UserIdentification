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
		NetworkRequest_EmbeddingCalculationSingleImage = 3,
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

		const NetworkRequestType cRequestType;
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

	protected:

		// submit specific payload
		void SubmitPayload() {

#ifdef _DEBUG
			if (mImage.size().width != mImage.size().height) {
				throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
			}
#endif
			// send image dimension
			pServerConn->SendUInt(mImage.size().width);

			// send image
			pServerConn->SendRGBImage(mImage);
		}

		// payload: quadratic(!) image
		cv::Mat mImage;

	};

	// embedding calculation
	class EmbeddingCalculationSingleImage : public NetworkRequest
	{
	public:
		EmbeddingCalculationSingleImage(
			io::TCPClient* server_conn,
			cv::Mat img
		) :
			NetworkRequest(server_conn, NetworkRequest_EmbeddingCalculationSingleImage),
			mImage(img)
		{
		}

	protected:

		// submit specific payload
		void SubmitPayload() {

#ifdef _DEBUG
			if (mImage.size().width != mImage.size().height) {
				throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
			}
#endif
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