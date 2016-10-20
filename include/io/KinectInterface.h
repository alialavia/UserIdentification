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

		// get deep copy of color frame
		void GetColorImageCopy(cv::Mat &dst);
		void GetDepthImageCopy(cv::Mat &dst);

	private:
		HRESULT ProcessColorFrame(IColorFrame *color_frame);
		HRESULT ProcessDepthFrame(IDepthFrame *depth_frame);
	
	public:
		// output settings
		const int mColorHeight;
		const int mColorWidth;
		const int mDepthHeight;
		const int mDepthWidth;

	private:
		std::mutex mSensorMutex;
		IKinectSensor* pSensor;
		IMultiSourceFrameReader* pSourceReader;

		// buffers containing latest data (refreshed with AcquireFrame)

		// color
		RGBQUAD *pColorImageBuffer;
		UINT mColorImageBufferLen = 0;
		int ColorImageStreamHeight = 0;
		int ColorImageStreamWidth = 0;

		// depth
		UINT16 *pDepthImageBuffer;
		UINT mDepthImageBufferLen = 0;
		int DepthImageStreamHeight = 0;
		int DepthImageStreamWidth = 0;

	};

};

#endif