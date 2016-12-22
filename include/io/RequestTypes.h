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
		NetworkRequest_ImageIdentificationAligned = 6,
		NetworkRequest_EmbeddingCollectionByIDAligned = 7,

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

	class ImageIdentification: public NetworkRequest
	{
	public:
		ImageIdentification(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images
		) :
			NetworkRequest(server_conn, NetworkRequest_ImageIdentification), mImages(images){}

		ImageIdentification(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images
		) :
			NetworkRequest(server_conn, NetworkRequest_ImageIdentification)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back((*images[i]).clone());
			}
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendImageBatchSquaredSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
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
		EmbeddingCollectionByID(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images,
			int user_id
		) :
			NetworkRequest(server_conn, NetworkRequest_EmbeddingCollectionByID),
			mUserID(user_id)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back((*images[i]).clone());
			}
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendUInt(mUserID);
			pServerConn->SendImageBatchSquaredSameSize(mImages);
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
		EmbeddingCollectionByName(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images,
			std::string user_name
		) :
			NetworkRequest(server_conn, NetworkRequest_EmbeddingCollectionByName),
			mUserName(user_name)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size();i++) {
				mImages.push_back((*images[i]).clone());
			}
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendString(mUserName);
			pServerConn->SendImageBatchSquaredSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		std::string mUserName;
	};


	class EmbeddingCallculation : public NetworkRequest
	{
	public:
		EmbeddingCallculation(
			io::TCPClient* server_conn,
			cv::Mat img
		) :
			NetworkRequest(server_conn, NetworkRequest_EmbeddingCalculation), mImage(img){}

	protected:
		// submit specific payload
		void SubmitPayload() {
			// send image
			pServerConn->SendRGBImageQuadratic(mImage);
		}

		// payload: quadratic(!) image
		cv::Mat mImage;
		int mUserID;
	};

	class ClassifierTraining : public NetworkRequest
	{
	public:
		ClassifierTraining(io::TCPClient* server_conn) :
			NetworkRequest(server_conn, NetworkRequest_ClassifierTraining){}
	protected:
		void SubmitPayload() {}
	};


	// ---------- online requests

	class ImageIdentificationAligned : public NetworkRequest
	{
	public:
		ImageIdentificationAligned(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images
		) :
			NetworkRequest(server_conn, NetworkRequest_ImageIdentificationAligned)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back((*images[i]).clone());
			}
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendImageBatchSquaredSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;

	};

	class EmbeddingCollectionByIDAligned : public NetworkRequest
	{
	public:
		EmbeddingCollectionByIDAligned(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images,
			int user_id
		) :
			NetworkRequest(server_conn, NetworkRequest_EmbeddingCollectionByIDAligned),
			mUserID(user_id)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back((*images[i]).clone());
			}
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendUInt(mUserID);	// user id
			pServerConn->SendImageBatchSquaredSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		int mUserID;
	};


}

#endif