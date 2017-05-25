#ifndef IO_REQUESTTYPES_H_
#define IO_REQUESTTYPES_H_
#include "ResponseTypes.h"
#include <opencv2/core/mat.hpp>
#include <io/Networking.h>
#include <unordered_set>
#include <tuple>
#include <chrono>

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
		// identification
		NetworkRequest_ImageIdentification = 1,
		NetworkRequest_ImageIdentificationAligned = 2,
		NetworkRequest_ImageIdentificationAlignedCS = 3,
		NetworkRequest_PartialImageIdentificationAligned = 4,
		// updates
		NetworkRequest_EmbeddingCollectionByID = 10,
		NetworkRequest_EmbeddingCollectionByIDRobust = 11,
		NetworkRequest_EmbeddingCollectionByIDAligned = 12,
		NetworkRequest_EmbeddingCollectionByIDAlignedRobust = 13,
		NetworkRequest_EmbeddingCollectionByName = 14,
		NetworkRequest_PartialUpdateAlignedRobust = 15,
		// utils
		NetworkRequest_EmbeddingCalculation = 20,
		NetworkRequest_ClassifierTraining = 21,
		NetworkRequest_ImageAlignment = 22,
		NetworkRequest_ProfilePictureUpdate = 23,
		NetworkRequest_GetProfilePictures = 24,
		// actions
		NetworkRequest_DropIdentificationSamples = 221,
		NetworkRequest_Ping = 222,
		NetworkRequest_Disconnect = 223
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
		__int64 mSubmitTime = 0;
	protected:
		io::TCPClient* pServerConn;
	protected:
		// submit specific payload (data size and data)
		virtual void SubmitPayload() = 0;

	};

	class ClientDisconnect : public NetworkRequest
	{
	public: ClientDisconnect(io::TCPClient* server_conn) :NetworkRequest(server_conn, NetworkRequest_Disconnect) {}
	protected: void SubmitPayload() {}
	};

	class Ping : public NetworkRequest
	{
	public: Ping(io::TCPClient* server_conn) :NetworkRequest(server_conn, NetworkRequest_Ping) {}
	protected: void SubmitPayload() {}
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
			pServerConn->SendImageBatchQuadraticSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;

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
			pServerConn->SendImageBatchQuadraticSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		std::string mUserName;
	};


	class EmbeddingCalculation : public NetworkRequest
	{
	public:
		EmbeddingCalculation(
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


	//class PartialImageIdentificationAligned : public NetworkRequest
	//{
	//public:
	//	PartialImageIdentificationAligned(
	//		io::TCPClient* server_conn,
	//		std::vector<cv::Mat*> images,
	//		std::vector<int> img_weights,
	//		int tracking_id
	//	) :
	//		NetworkRequest(server_conn, NetworkRequest_PartialImageIdentificationAligned),
	//		mImageWeights(img_weights),
	//		mTrackingID(tracking_id)
	//	{
	//		// make deep copy of images
	//		for (size_t i = 0; i < images.size(); i++) {
	//			mImages.push_back((*images[i]).clone());
	//		}
	//	}
	//	PartialImageIdentificationAligned(
	//		io::TCPClient* server_conn,
	//		std::vector<cv::Mat> images,
	//		std::vector<int> img_weights,
	//		int tracking_id
	//	) :
	//		NetworkRequest(server_conn, NetworkRequest_PartialImageIdentificationAligned),
	//		mImageWeights(img_weights),
	//		mTrackingID(tracking_id)
	//	{
	//		// make deep copy of images
	//		for (size_t i = 0; i < images.size(); i++) {
	//			mImages.push_back(images[i].clone());
	//		}
	//	}

	//protected:

	//	// submit specific payload
	//	void SubmitPayload() {
	//		// tracking id
	//		pServerConn->SendUInt(mTrackingID);
	//		// images
	//		pServerConn->SendImageBatchQuadraticSameSize(mImages);
	//		// weights
	//		pServerConn->SendUCharArray(mImageWeights);
	//	}

	//	// payload: quadratic(!) images of same size
	//	std::vector<cv::Mat> mImages;
	//	std::vector<int> mImageWeights;
	//	int mTrackingID;

	//};

	class PartialImageIdentificationAligned : public NetworkRequest
	{
	public:
		PartialImageIdentificationAligned(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images,
			std::vector<std::tuple<int, int>> img_poses,
			int tracking_id
		) :
			NetworkRequest(server_conn, NetworkRequest_PartialImageIdentificationAligned),
			mImagePoses(img_poses),
			mTrackingID(tracking_id)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back((*images[i]).clone());
			}
		}
		PartialImageIdentificationAligned(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images,
			std::vector<std::tuple<int, int>> img_poses,
			int tracking_id
		) :
			NetworkRequest(server_conn, NetworkRequest_PartialImageIdentificationAligned),
			mImagePoses(img_poses),
			mTrackingID(tracking_id)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back(images[i].clone());
			}
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			// tracking id
			pServerConn->SendUInt(mTrackingID);
			// images
			pServerConn->SendImageBatchQuadraticSameSize(mImages);
			// weights
			for (auto& elem : mImagePoses) {
				// pitch
				pServerConn->SendChar(std::get<0>(elem));
				// yaw
				pServerConn->SendChar(std::get<1>(elem));
			}
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		std::vector<std::tuple<int,int>> mImagePoses;
		int mTrackingID;

	};

	// closed set image identification
	class ImageIdentificationAlignedCS : public NetworkRequest
	{
	public:
		ImageIdentificationAlignedCS(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images,
			std::unordered_set<int> ids
		) :
			NetworkRequest(server_conn, NetworkRequest_ImageIdentificationAlignedCS)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back((*images[i]).clone());
			}
			mIDs = ids;
		}
		ImageIdentificationAlignedCS(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images,
			std::unordered_set<int> ids
		) :
			NetworkRequest(server_conn, NetworkRequest_ImageIdentificationAlignedCS)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back(images[i].clone());
			}
			mIDs = ids;
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendUInt(mIDs.size());
			for (auto const& id : mIDs) {
				pServerConn->SendUInt(id);
			}
			pServerConn->SendImageBatchQuadraticSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		// possible ids
		std::unordered_set<int> mIDs;

	};

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
		ImageIdentificationAligned(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images
		) :
			NetworkRequest(server_conn, NetworkRequest_ImageIdentificationAligned)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back(images[i].clone());
			}
		}

	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendImageBatchQuadraticSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;

	};


	class EmbeddingCollectionByID : public NetworkRequest
	{
	public:
		EmbeddingCollectionByID(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images,
			int user_id,
			// specify sub type (prealigned, robust)
			NetworkRequestType sub_type = NetworkRequest_EmbeddingCollectionByID
		) :
			NetworkRequest(server_conn, sub_type),
			mUserID(user_id)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back((*images[i]).clone());
			}
		}
		EmbeddingCollectionByID(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images,
			int user_id,
			NetworkRequestType sub_type = NetworkRequest_EmbeddingCollectionByID
		) :
			NetworkRequest(server_conn, sub_type),
			mUserID(user_id)
		{
			mImages = images;
		}
	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendUInt(mUserID);	// user id
			pServerConn->SendImageBatchQuadraticSameSize(mImages);
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		// Todo: careful! Int sent as Uint
		int mUserID;
	};

	//class PartialUpdateAligned : public NetworkRequest
	//{
	//public:
	//	PartialUpdateAligned(
	//		io::TCPClient* server_conn,
	//		std::vector<cv::Mat*> images,
	//		std::vector<int> weights,
	//		int user_id,
	//		// specify sub type (prealigned, robust)
	//		NetworkRequestType sub_type = NetworkRequest_PartialUpdateAlignedRobust
	//	) :
	//		NetworkRequest(server_conn, sub_type),
	//		mImageWeights(weights),
	//		mUserID(user_id)
	//	{
	//		// make deep copy of images
	//		for (size_t i = 0; i < images.size(); i++) {
	//			mImages.push_back((*images[i]).clone());
	//		}
	//	}
	//	PartialUpdateAligned(
	//		io::TCPClient* server_conn,
	//		std::vector<cv::Mat> images,
	//		std::vector<int> weights,
	//		int user_id,
	//		NetworkRequestType sub_type = NetworkRequest_PartialUpdateAlignedRobust
	//	) :
	//		NetworkRequest(server_conn, sub_type),
	//		mImageWeights(weights),
	//		mUserID(user_id)
	//	{
	//		mImages = images;
	//	}
	//protected:

	//	// submit specific payload
	//	void SubmitPayload() {
	//		pServerConn->SendUInt(mUserID);	// user id
	//		// images
	//		pServerConn->SendImageBatchQuadraticSameSize(mImages);
	//		// weights
	//		pServerConn->SendUCharArray(mImageWeights);
	//	}

	//	// payload: quadratic(!) images of same size
	//	std::vector<cv::Mat> mImages;
	//	std::vector<int> mImageWeights;
	//	// Todo: careful! Int sent as Uint
	//	int mUserID;
	//};


	class PartialUpdateAligned : public NetworkRequest
	{
	public:
		PartialUpdateAligned(
			io::TCPClient* server_conn,
			std::vector<cv::Mat*> images,
			std::vector<std::tuple<int, int>> img_poses,
			int user_id,
			// specify sub type (prealigned, robust)
			NetworkRequestType sub_type = NetworkRequest_PartialUpdateAlignedRobust
		) :
			NetworkRequest(server_conn, sub_type),
			mImagePoses(img_poses),
			mUserID(user_id)
		{
			// make deep copy of images
			for (size_t i = 0; i < images.size(); i++) {
				mImages.push_back((*images[i]).clone());
			}
		}
		PartialUpdateAligned(
			io::TCPClient* server_conn,
			std::vector<cv::Mat> images,
			std::vector<std::tuple<int, int>> img_poses,
			int user_id,
			NetworkRequestType sub_type = NetworkRequest_PartialUpdateAlignedRobust
		) :
			NetworkRequest(server_conn, sub_type),
			mImagePoses(img_poses),
			mUserID(user_id)
		{
			mImages = images;
		}
	protected:

		// submit specific payload
		void SubmitPayload() {
			// user id
			pServerConn->SendUInt(mUserID);	
			// images
			pServerConn->SendImageBatchQuadraticSameSize(mImages);
			// weights
			for (auto& elem : mImagePoses) {
				// pitch
				pServerConn->SendChar(std::get<0>(elem));
				// yaw
				pServerConn->SendChar(std::get<1>(elem));
			}
		}

		// payload: quadratic(!) images of same size
		std::vector<cv::Mat> mImages;
		std::vector<std::tuple<int, int>> mImagePoses;
		// Todo: careful! Int sent as Uint
		int mUserID;
	};


	class ImageAlignment : public NetworkRequest
	{
	public:
		ImageAlignment(
			io::TCPClient* server_conn,
			cv::Mat img
		) :
			NetworkRequest(server_conn, NetworkRequest_ImageAlignment), mImage(img)
		{
		}
	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendRGBImageQuadratic(mImage);
		}

		// payload: quadratic(!) image
		cv::Mat mImage;
	};

	class ProfilePictureUpdate : public NetworkRequest
	{
	public:
		ProfilePictureUpdate(
			io::TCPClient* server_conn,
			int user_id,
			cv::Mat img
		) :
			NetworkRequest(server_conn, NetworkRequest_ProfilePictureUpdate), mUserID(user_id), mImage(img)
		{
		}
	protected:

		// submit specific payload
		void SubmitPayload() {
			pServerConn->SendUInt(mUserID);	// user id
			pServerConn->SendRGBImageQuadratic(mImage);
		}

		// Todo: careful! Int sent as Uint
		int mUserID;
		// payload: quadratic(!) image
		cv::Mat mImage;
	};

	class GetProfilePictures : public NetworkRequest
	{
	public: GetProfilePictures(io::TCPClient* server_conn) :NetworkRequest(server_conn, NetworkRequest_GetProfilePictures){}
	protected: void SubmitPayload() {}
	};

	class DropIdentificationSamples : public NetworkRequest
	{
	public: DropIdentificationSamples(io::TCPClient* server_conn, int tracking_id) :NetworkRequest(server_conn, NetworkRequest_DropIdentificationSamples), mTrackingID(tracking_id){}
	protected: 
		void SubmitPayload()
		{
			pServerConn->SendUInt(mTrackingID);
		}
		int mTrackingID;
	};

	


}

#endif