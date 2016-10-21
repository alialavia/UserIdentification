#include <string>
#include <iostream>
#include <base/UserIdentification.h>
#include <io/KinectInterface.h>

// pointer safe release
#include <strsafe.h>

// OpenCV
#include <opencv2/highgui/highgui.hpp>

// image processing
#include "opencv2/imgproc.hpp"

// Kinect SDK 2
#include <Kinect.h>

//MultiSourceFrameReader will align to the slowest framerate of any subscribed source. In lowlight scenarios,
//the color stream may drop to 15 FPS.If this happens, and MultiSourceFrameReader is subscribed 
//to color as one of its subscribed sources, the rate of delivered frames will drop to 15 FPS for the entire instance of this MultiSourceFrameReader

using namespace io;

KinectSensorMultiSource::KinectSensorMultiSource() : 
	pSourceReader(NULL), 
	mColorWidth(1920),
	mColorHeight(1080),
	mDepthImageWidth(480),
	mDepthImageHeight(320),
	pColorImageBuffer(NULL),
	pDepthBuffer(NULL)
{

}

// Careful! Extracted images will be deleted if the sensor streamer runs out of scope
KinectSensorMultiSource::~KinectSensorMultiSource() {
	Close();

	// cleanup
	if (pColorImageBuffer != NULL) {
		delete[] pColorImageBuffer;
		pColorImageBuffer = NULL;
	}
}

void KinectSensorMultiSource::Close() {
	// shut down streams
	// close the Kinect Sensor
	if (pSensor)
	{
		pSensor->Close();
	}

	SafeRelease(pSensor);

	// release reader
	SafeRelease(pSourceReader);
}

HRESULT KinectSensorMultiSource::Open() {

	HRESULT hr;

	// connect to sensor
	hr = GetDefaultKinectSensor(&pSensor);

	if (FAILED(hr)) {
		std::cout << "Error : GetDefaultKinectSensor" << std::endl;
		return hr;
	}

	// open sensor
	hr = pSensor->Open();
	if (FAILED(hr)) {
		std::cout << "Error : IKinectSensor::Open()" << std::endl;
		return hr;
	}

	// open reader
	hr = pSensor->OpenMultiSourceFrameReader(
		FrameSourceTypes::FrameSourceTypes_Depth |
		FrameSourceTypes::FrameSourceTypes_Color |
		FrameSourceTypes::FrameSourceTypes_Infrared |
		FrameSourceTypes::FrameSourceTypes_Body |
		FrameSourceTypes::FrameSourceTypes_BodyIndex,
		&pSourceReader);

	if (FAILED(hr)) {
		std::cout << "Error : OpenMultiSourceFrameReader()" << std::endl;
		return hr;
	}

	return S_OK;
}

HRESULT KinectSensorMultiSource::AcquireFrame() {

	if (!pSensor) {
		return E_FAIL;
	}

	HRESULT hr = E_PENDING;

	// allocate frames
	IMultiSourceFrame* p_multisource_frame = NULL;
	IDepthFrame* p_depth_frame = NULL;
	IColorFrame* p_color_frame = NULL;
	IInfraredFrame* p_infrared_frame = NULL;
	IBodyFrame* p_body_frame = NULL;
	IBodyIndexFrame* p_bodyindex_frame = NULL;

	// get multiframe
	hr = pSourceReader->AcquireLatestFrame(&p_multisource_frame);

	// TODO: DOES AcquireLatestFrame ONLY GIVE NEW FRAMES OR DO WE NEED TO CHECK
	// IF FPS > FPS_kINECT:=30
	if (SUCCEEDED(hr))
	{
		// color
		if (SUCCEEDED(hr))
		{
			IColorFrameReference* cfr = NULL;

			hr = p_multisource_frame->get_ColorFrameReference(&cfr);
			if (SUCCEEDED(hr))
			{
				hr = cfr->AcquireFrame(&p_color_frame);
			}
			SafeRelease(cfr);
		}
		// depth
		if (SUCCEEDED(hr))
		{
			IDepthFrameReference* dfr = NULL;

			hr = p_multisource_frame->get_DepthFrameReference(&dfr);
			if (SUCCEEDED(hr))
			{
				hr = dfr->AcquireFrame(&p_depth_frame);
			}
			SafeRelease(dfr);
		}
		// infrared
		if (SUCCEEDED(hr))
		{
			IInfraredFrameReference* ifr = NULL;

			hr = p_multisource_frame->get_InfraredFrameReference(&ifr);
			if (SUCCEEDED(hr))
			{
				hr = ifr->AcquireFrame(&p_infrared_frame);
			}
			SafeRelease(ifr);
		}
		// body
		if (SUCCEEDED(hr))
		{
			IBodyFrameReference* bfr = NULL;

			hr = p_multisource_frame->get_BodyFrameReference(&bfr);
			if (SUCCEEDED(hr))
			{
				hr = bfr->AcquireFrame(&p_body_frame);
			}
			SafeRelease(bfr);
		}

		// body index
		if (SUCCEEDED(hr))
		{
			IBodyIndexFrameReference* bifr = NULL;

			hr = p_multisource_frame->get_BodyIndexFrameReference(&bifr);
			if (SUCCEEDED(hr))
			{
				hr = bifr->AcquireFrame(&p_bodyindex_frame);
			}
			SafeRelease(bifr);
		}
	}

	// process frames if everything went smoothly
	if (SUCCEEDED(hr)) {
		mSensorMutex.lock();	// class internal data manipulations
		// color
		ProcessColorFrame(p_color_frame, ColorImageStreamHeight, ColorImageStreamWidth, pColorImageBuffer, mColorImageBufferLen);
		// depth
		ProcessDepthFrame(p_depth_frame, DepthStreamHeight, DepthStreamWidth, pDepthBuffer, mDepthBufferLen);
		mSensorMutex.unlock();
	}

	// release multiframe
	SafeRelease(p_multisource_frame);

	// release extracted frames - unblocking other readers
	SafeRelease(p_depth_frame);
	SafeRelease(p_color_frame);
	SafeRelease(p_infrared_frame);
	SafeRelease(p_body_frame);
	SafeRelease(p_bodyindex_frame);

	// return status
	return hr;
}

// --------------------- PROCESSING FUNCTIONS

HRESULT KinectSensorMultiSource::ProcessBodyFrame(IBodyFrame *body_frame) {

	IFrameDescription *frameDesc = nullptr;
	HRESULT hr = E_FAIL;

	// BODY_COUNT = 6
	IBody* ppBodies[BODY_COUNT] = { 0 };

	hr = body_frame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);

	// readin body buffer
	if (SUCCEEDED(hr)) {


	}


	// free memory
	for (int i = 0; i < _countof(ppBodies); ++i)
	{
		SafeRelease(ppBodies[i]);
	}

	return hr;
}

HRESULT KinectSensorMultiSource::ProcessColorFrame(IColorFrame* color_frame, int &height, int &width,  RGBQUAD* &buffer, UINT &buffer_len) {

	IFrameDescription *frameDesc = nullptr;
	HRESULT hr = E_FAIL;

	// output parameters
	ColorImageFormat targetFormat = ColorImageFormat_Bgra;

	// current information
	ColorImageFormat imageFormat;

	// current image buffer length
	UINT nBufferLen;

	hr = color_frame->get_FrameDescription(&frameDesc);

	// readin image buffer
	if (SUCCEEDED(hr)) {
		
		// get stream height and width
		if (
			SUCCEEDED(frameDesc->get_Height(&height)) &&
			SUCCEEDED(frameDesc->get_Width(&width)) &&
			SUCCEEDED(color_frame->get_RawColorImageFormat(&imageFormat))
			)
		{
			// allocate buffer
			nBufferLen = ColorImageStreamHeight * ColorImageStreamWidth * sizeof(RGBQUAD);

			// allocate more memory if necessary
			if (nBufferLen > buffer_len) {
				if (buffer != NULL) {
					delete[] buffer;
				}
				buffer = new RGBQUAD[nBufferLen];
				buffer_len = nBufferLen;
			}

			// no type conversion
			if (imageFormat == targetFormat)
			{	
				// TODO: check if buffer ist still available once frame is released (after polling). SHOULD NOT!
				// hr = color_frame->AccessRawUnderlyingBuffer(&nBufferLen, reinterpret_cast<BYTE**>(&pColorImageBuffer));
				hr = color_frame->CopyRawFrameDataToArray(nBufferLen, reinterpret_cast<BYTE*>(buffer));
			}
			// do color format conversion
			else
			{
				hr = color_frame->CopyConvertedFrameDataToArray(nBufferLen, reinterpret_cast<BYTE*>(buffer), targetFormat);
			}
		}
	}

	SafeRelease(frameDesc);
	// DO NOT RELEASE FRAME HERE! THIS IS A PURE PROCESSING FUNCTION

	return hr;
}

HRESULT KinectSensorMultiSource::ProcessDepthFrame(IDepthFrame *depth_frame, int &height, int &width, UINT16* &buffer, UINT &buffer_len) {

	HRESULT hr;


	IFrameDescription* frameDesc = nullptr;
	USHORT nDepthMinReliableDistance = 0;
	USHORT nDepthMaxDistance = 0;

	// current image buffer length
	UINT nBufferLen;

	UINT nBufferSize = 0;

	hr = depth_frame->get_FrameDescription(&frameDesc);

	// readin image buffer
	if (SUCCEEDED(hr)) {

		// get stream height and width
		if (
			SUCCEEDED(frameDesc->get_Height(&height)) &&
			SUCCEEDED(frameDesc->get_Width(&width)) &&
			SUCCEEDED(depth_frame->get_DepthMinReliableDistance(&nDepthMinReliableDistance)) &&
			SUCCEEDED(depth_frame->get_DepthMaxReliableDistance(&nDepthMaxDistance))
			)
		{
			// allocate buffer
			nBufferLen = height * width * sizeof(UINT16);

			// allocate more memory if necessary
			if (nBufferLen > buffer_len) {
				if (buffer != NULL) {
					delete[] buffer;
				}
				buffer = new UINT16[nBufferLen];
				buffer_len = nBufferLen;
			}

			// BYTE intensity = static_cast<BYTE>((depth >= nMinDepth) && (depth <= nMaxDepth) ? (depth % 256) : 0);
			//  reinterpret_cast<BYTE*>(pColorImageBuffer)
			hr = depth_frame->CopyFrameDataToArray(buffer_len, buffer);
		}
	}

	SafeRelease(frameDesc);
	// DO NOT RELEASE FRAME HERE! THIS IS A PURE PROCESSING FUNCTION

	return hr;
}

void KinectSensorMultiSource::GetColorImageCopy(cv::Mat &dst) {
	mSensorMutex.lock();

	// convert to mat
	// 8bit (0-256) BGRA (Blue, Green, Red, Alpha)
	cv::Mat cv_img(ColorImageStreamHeight, ColorImageStreamWidth, CV_8UC4, reinterpret_cast<void*>(pColorImageBuffer));

	// alternatively copy directly to cv image
	// TODO: check if this could be useful
	// pFrame->CopyFrameDataToArray(iWidth * iHeight,reinterpret_cast<UINT16*>(cv_img.data));

	cv::Mat resized;

	// make a copy and resize to output format
	// cv_img still points to color buffer!
	cv::resize(cv_img, resized, cv::Size(mColorWidth, mColorHeight));
	// alternatively, if dimension stays the same: dst = cv_img.copyTo(dst) - copies everything, not only header (in contrast to .clone())

	// set pointer of dst to resized image
	dst = resized;

	mSensorMutex.unlock();
}


void KinectSensorMultiSource::GetColorImage(cv::Mat &dst) {
	cv::Mat cv_img(ColorImageStreamHeight, ColorImageStreamWidth, CV_8UC4, reinterpret_cast<void*>(pColorImageBuffer));
	dst = cv_img;
}


void KinectSensorMultiSource::GetDepthImageCopy(cv::Mat &dst) {
	mSensorMutex.lock();
	// tmp
	cv::Mat cv_img(DepthStreamHeight, DepthStreamWidth, CV_16U, pDepthBuffer);
	cv::Mat resized;

	cv::resize(cv_img, resized, cv::Size(mDepthImageWidth, mDepthImageHeight));
	//double scale = 255.0 / (nDepthMaxReliableDistance - nDepthMinReliableDistance);
	//double scale = 255.0 / 5;
	//resized.convertTo(resized, CV_8UC1, scale);

	dst = resized;
	mSensorMutex.unlock();
}
