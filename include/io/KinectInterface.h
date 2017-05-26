#ifndef IO_KINECTINTERFACE_H_
#define IO_KINECTINTERFACE_H_

// MS Error codes
#include <strsafe.h>

// mutex
#include <mutex>
#include <Windows.h>

#include <iostream>

#include <base/UserIdentification.h>

// OpenCV
#include <opencv2/video/tracking.hpp>


#include <Kinect.Face.h>
#include <opencv2/highgui.hpp>


// forward declarations
struct IDepthFrame;
struct IColorFrame;
struct IInfraredFrame;
struct IBodyFrame;;
struct IBodyIndexFrame;
struct IMultiSourceFrameReader;
struct IKinectSensor;
struct IBody;
struct IFaceFrameSource;
struct IFaceFrameReader;
struct IFaceFrame;
struct ICoordinateMapper;

struct FaceData
{
	bool tracked = false;
	RectI boundingBox = { 0 };
	RectI boundingBoxIR = { 0 };
	PointF Points[FacePointType::FacePointType_Count];
	PointF PointsIR[FacePointType::FacePointType_Count];
	Vector4 Rotation;
	DetectionResult Properties[FaceProperty::FaceProperty_Count];
};

namespace cv
{
	class Mat;
}

namespace io
{
	class KinectSensorMultiSource
	{
	public:
		KinectSensorMultiSource();
		~KinectSensorMultiSource();

		HRESULT Open(int timeout = 5000);
		void Close();

		HRESULT AcquireFrame();

		// get deep copy of data
		void GetImageCopyBGRA(cv::Mat& dst) const;
		void GetImageCopyBGR(cv::Mat& dst) const;
		void GetImageCopyBGRSubtracted(cv::Mat& dst) const;
		void GetImageCopyDepth(cv::Mat& dst) const;
		void GetImageCopyDepth8UThresholded(cv::Mat& dst) const;
		void GetImageCopyBodyIndex(cv::Mat& dst) const;
		void GetImageCopyBodyIndexColored(cv::Mat& dst) const;
		void GetImageCopyBGRSkeleton(cv::Mat& dst) const;

		// link to current data
		void GetImageRGBA(cv::Mat& dst);

		HRESULT GetSensorReference(IKinectSensor* &s);
		IBody** GetBodyDataReference();
		FaceData* GetFaceDataReference();

		TIMESPAN GetBodyTimeStamp();

	private:
		HRESULT ProcessColorFrame(IColorFrame* color_frame, int& height, int& width, RGBQUAD* & buffer, UINT& buffer_len) const;
		HRESULT ProcessDepthFrame(IDepthFrame* depth_frame, int& height, int& width, UINT16* & buffer, UINT& buffer_len) const;
		HRESULT ProcessBodyFrame(IBodyFrame* body_frame);
		HRESULT ProcessFaceFrames(IFaceFrame* pFaceFrame[NR_USERS]);
		HRESULT ProcessBodyIndexFrame(IBodyIndexFrame* index_frame, int& height, int& width, BYTE* & buffer, UINT& buffer_len);
		bool IsValidFace(const FaceData &face) const;

	public:
		// output settings
		const int mColorHeight;
		const int mColorWidth;
		const int mDepthImageHeight;
		const int mDepthImageWidth;

	private:
		// sensor handles
		mutable std::mutex mSensorMutex;
		IKinectSensor* pSensor;
		ICoordinateMapper* m_pCoordinateMapper;

		// multisource reader
		IMultiSourceFrameReader* pSourceReader;
		// face source/reader
		IFaceFrameSource* m_pFaceFrameSources[NR_USERS];
		IFaceFrameReader* m_pFaceFrameReaders[NR_USERS];

		// -------- buffers containing latest data (refreshed with AcquireFrame)

		// color
		RGBQUAD* pColorImageBuffer;
		UINT mColorImageBufferLen = 0;
		int ColorImageStreamHeight = 0;
		int ColorImageStreamWidth = 0;
		// depth
		UINT16* pDepthBuffer;
		UINT mDepthBufferLen = 0;
		int DepthStreamHeight = 0;
		int DepthStreamWidth = 0;
		// bodies
		IBody* ppBodies[NR_USERS] = {0};
		Vector4 mFloor = {0};
		TIMESPAN mBodyTimeStamp;
		// faces
		FaceData mFaces[NR_USERS];
		// body index
		BYTE* pBodyIndexBuffer;
		UINT mBodyIndexBufferLen = 0;
		int BodyIndexStreamHeight = 0;
		int BodyIndexStreamWidth = 0;
	};
};

#endif
