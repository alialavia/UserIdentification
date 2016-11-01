#ifndef IO__kinectinterface
#define IO__kinectinterface

// MS Error codes
#include <strsafe.h>

// mutex
#include <mutex>

#include <Windows.h>

// OpenCV
#include <opencv2/core/core.hpp>

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


#define NR_USERS 6

namespace cv{
	class Mat;
}

namespace io
{
	class KinectSensorMultiSource {

	public:
		KinectSensorMultiSource();
		~KinectSensorMultiSource();

		HRESULT Open();
		void Close();

		HRESULT AcquireFrame();

		// get deep copy of data
		void GetImageCopyRGBA(cv::Mat &dst);
		void GetImageCopyRGB(cv::Mat &dst);
		void GetImageCopyDepth(cv::Mat &dst);
		// link to current data
		void GetImageRGBA(cv::Mat &dst);

	private:
		HRESULT ProcessColorFrame(IColorFrame* color_frame, int &height, int &width, RGBQUAD* &buffer, UINT &buffer_len);
		HRESULT ProcessDepthFrame(IDepthFrame *depth_frame, int &height, int &width, UINT16* &buffer, UINT &buffer_len);
		HRESULT ProcessBodyFrame(IBodyFrame *body_frame);
		HRESULT ProcessFaces(IFaceFrame* pFaceFrame[NR_USERS]);
	
	public:
		// output settings
		const int mColorHeight;
		const int mColorWidth;
		const int mDepthImageHeight;
		const int mDepthImageWidth;

	private:
		std::mutex mSensorMutex;
		IKinectSensor* pSensor;
		IMultiSourceFrameReader* pSourceReader;

		// face source/reader
		IFaceFrameSource* m_pFaceFrameSources[NR_USERS];
		IFaceFrameReader* m_pFaceFrameReaders[NR_USERS];

		// -------- buffers containing latest data (refreshed with AcquireFrame)

		// color
		RGBQUAD *pColorImageBuffer;
		UINT mColorImageBufferLen = 0;
		int ColorImageStreamHeight = 0;
		int ColorImageStreamWidth = 0;

		// depth
		UINT16 *pDepthBuffer;
		UINT mDepthBufferLen = 0;
		int DepthStreamHeight = 0;
		int DepthStreamWidth = 0;

		// bodies
		IBody* ppBodies[NR_USERS] = { 0 };


	};

};

#endif