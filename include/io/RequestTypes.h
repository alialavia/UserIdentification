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
		NetworkRequest_ImageIdentification = 1,
		NetworkRequest_EmbeddingCollectionByID = 2,
		NetworkRequest_EmbeddingCollectionByName = 3,
		NetworkRequest_EmbeddingCalculation = 4,
		NetworkRequest_ClassifierTraining = 5,
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
		void SendImageBatchSquaredSameSize(const std::vector<cv::Mat> &images) const
		{
#ifdef _DEBUG
			// check image dimensions
			int img_size = images[0].size().width;
			for (size_t i = 0; i<images.size(); i++)
			{
				if (images[i].size().width != img_size || images[i].size().height != img_size) {
					throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
				}
			}
#endif
			if (images.size() == 0)
			{
				throw std::invalid_argument("No data to send.");
			}

			// send nr images
			pServerConn->SendInt(images.size());

			// send image dimension
			pServerConn->SendUInt(images[0].size().width);

			// send images
			for (size_t i = 0; i<images.size(); i++)
			{
				pServerConn->SendRGBImage(images[i]);
			}
		}

		void SendImageBatchSameSize(const std::vector<cv::Mat> &images) const
		{
#ifdef _DEBUG
			// check image dimensions
			int img_width = images[0].size().width;
			int img_height = images[0].size().height;
			for (size_t i = 0; i<images.size(); i++)
			{
				if (images[i].size().width != img_width || images[i].size().height != img_height) {
					throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
				}
			}
#endif
			if (images.size() == 0)
			{
				throw std::invalid_argument("No data to send.");
			}

			// send nr images
			pServerConn->SendInt(images.size());

			// send image dimension
			pServerConn->SendUInt(images[0].size().height);
			pServerConn->SendUInt(images[0].size().width);

			// send images
			for (size_t i = 0; i<images.size(); i++)
			{
				pServerConn->SendRGBImage(images[i]);
			}
		}

		void SendImageBatchSquared(const std::vector<cv::Mat> &images) const
		{
#ifdef _DEBUG
			// check image dimensions
			for (size_t i = 0; i<images.size(); i++)
			{
				if (images[i].size().width != images[i].size().height) {
					throw std::invalid_argument("Invalid image dimensions - Image must be quadratic!");
				}
			}
#endif
			if (images.size() == 0)
			{
				throw std::invalid_argument("No data to send.");
			}

			// send nr images
			pServerConn->SendInt(images.size());

			// send images
			for (size_t i = 0; i<images.size(); i++)
			{
				// send image dimension
				pServerConn->SendUInt(images[i].size().width);

				// send image
				pServerConn->SendRGBImage(images[i]);
			}
		}

		void SendImageBatch(const std::vector<cv::Mat> &images) const
		{
			if (images.size() == 0)
			{
				throw std::invalid_argument("No data to send.");
			}

			// send nr images
			pServerConn->SendInt(images.size());

			// send images
			for (size_t i = 0; i<images.size(); i++)
			{
				// send image width
				pServerConn->SendUInt(images[i].size().width);

				// send image height
				pServerConn->SendUInt(images[i].size().height);

				// send image
				pServerConn->SendRGBImage(images[i]);
			}
		}

	protected:
		// submit specific payload (data size and data)
		virtual void SubmitPayload() = 0;

	};

	// -----------------------------------------
	//		ADD CUSTOM IMPLEMENTATIONS HERE
	// -----------------------------------------
	// - add NetworkRequestType
	// - add Generation in Factory

	class ImageIdentificationRequest: public NetworkRequest
	{
	public:
		ImageIdentificationRequest(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images
		) :
			NetworkRequest(server_conn, NetworkRequest_ImageIdentification),
			mImages(images)
		{
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			SendImageBatchSquaredSameSize(mImages);
		}

		// payload: quadratic(!) image
		std::vector<cv::Mat> mImages;

	};

	class EmbeddingCollectionByID : public NetworkRequest
	{
	public:
		EmbeddingCollectionByID(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images,
			int user_id
		) :
			NetworkRequest(server_conn, NetworkRequest_EmbeddingCollectionByID),
			mImages(images),
			mUserID(user_id)
		{
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendUInt(mUserID);
			SendImageBatchSquaredSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		int mUserID;
	};

	class EmbeddingCollectionByName : public NetworkRequest
	{
	public:
		EmbeddingCollectionByName(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images,
			std::string user_name
		) :
			NetworkRequest(server_conn, NetworkRequest_EmbeddingCollectionByName),
			mImages(images),
			mUserName(user_name)
		{
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendString(mUserName);
			SendImageBatchSquaredSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		std::string mUserName;
	};


}

#endif