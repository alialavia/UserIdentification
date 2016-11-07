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

namespace cv
{
	class Mat;
}

namespace io
{
	struct Face
	{
		RectI faceBox = {0};
		PointF facePoints[FacePointType::FacePointType_Count];
		Vector4 faceRotation;
		DetectionResult faceProperties[FaceProperty::FaceProperty_Count];
	};

	class KinectSensorMultiSource
	{
	public:
		KinectSensorMultiSource();
		~KinectSensorMultiSource();

		HRESULT Open(int timeout = 3000);
		void Close();

		HRESULT AcquireFrame();

		// get deep copy of data
		void GetImageCopyRGBA(cv::Mat& dst) const;
		void GetImageCopyRGB(cv::Mat& dst) const;
		void GetImageCopyDepth(cv::Mat& dst) const;
		void GetImageCopyBodyIndexColored(cv::Mat& dst) const;
		void GetImageCopyRGBSkeleton(cv::Mat& dst) const;

		// link to current data
		void GetImageRGBA(cv::Mat& dst);

		HRESULT GetSensorReference(IKinectSensor* &s);
		IBody** GetBodyDataReference();


		/*
		void drawFaces(cv::Mat& dst)
		{
			for (int iFace = 0; iFace < NR_USERS; ++iFace)
			{
				cv::rectangle(
					dst,
					cv::Point(Faces[iFace].faceBox.Bottom, Faces[iFace].faceBox.Right),
					cv::Point(Faces[iFace].faceBox.Bottom, Faces[iFace].faceBox.Left),
					cv::Scalar(255, 255, 255)
				);
			}

			cv::imshow("Color image", dst);
			cv::waitKey(3);
		}

		void printFaces()
		{
			// iterate through each face reader
			for (int iFace = 0; iFace < NR_USERS; ++iFace)
			{
				if (Faces[iFace].faceBox.Bottom > 0 && Faces[iFace].faceBox.Top > 0)
				{
					std::cout << "Face " << iFace << " - " << Faces[iFace].faceBox.Bottom << " - " << Faces[iFace].faceBox.Top << "\n";
				}
			}
		}
		*/

	private:
		HRESULT ProcessColorFrame(IColorFrame* color_frame, int& height, int& width, RGBQUAD* & buffer, UINT& buffer_len) const;
		HRESULT ProcessDepthFrame(IDepthFrame* depth_frame, int& height, int& width, UINT16* & buffer, UINT& buffer_len) const;
		HRESULT ProcessBodyFrame(IBodyFrame* body_frame);
		HRESULT ProcessFaceFrames(IFaceFrame* pFaceFrame[NR_USERS]);
		HRESULT ProcessBodyIndexFrame(IBodyIndexFrame* index_frame, int& height, int& width, BYTE* & buffer, UINT& buffer_len);

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
		// faces
		Face Faces[NR_USERS];
		// body index
		BYTE* pBodyIndexBuffer;
		UINT mBodyIndexBufferLen = 0;
		int BodyIndexStreamHeight = 0;
		int BodyIndexStreamWidth = 0;
	};
};

#endif
