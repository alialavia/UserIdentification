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
#include <Kinect.Face.h>

#include <windows.h>

//MultiSourceFrameReader will align to the slowest framerate of any subscribed source. In lowlight scenarios,
//the color stream may drop to 15 FPS.If this happens, and MultiSourceFrameReader is subscribed 
//to color as one of its subscribed sources, the rate of delivered frames will drop to 15 FPS for the entire instance of this MultiSourceFrameReader

using namespace io;


KinectSensorMultiSource::KinectSensorMultiSource(): 
	pSourceReader(NULL), 
	mColorWidth(1920),
	mColorHeight(1080),
	mDepthImageWidth(480),
	mDepthImageHeight(320),
	pColorImageBuffer(NULL),
	pDepthBuffer(NULL)
{
	// init face frame source and readers
	for (int i = 0; i < NR_USERS; i++)
	{
		m_pFaceFrameSources[i] = nullptr;
		m_pFaceFrameReaders[i] = nullptr;
	}

}

// Careful! Extracted images will be deleted if the sensor streamer runs out of scope
KinectSensorMultiSource::~KinectSensorMultiSource() {
	Close();

	// cleanup buffers

	// color
	if (pColorImageBuffer != nullptr) {
		delete[] pColorImageBuffer;
		pColorImageBuffer = nullptr;
	}
	// depth
	if (pDepthBuffer != nullptr) {
		delete[] pDepthBuffer;
		pDepthBuffer = nullptr;
	}
	// body data
	for (int i = 0; i < _countof(ppBodies); ++i)
	{
		SafeRelease(ppBodies[i]);
	}

}

void KinectSensorMultiSource::Close() {

	// release reader
	SafeRelease(pSourceReader);

	// face
	for (int i = 0; i < NR_USERS; i++)
	{
		SafeRelease(m_pFaceFrameSources[i]);
		SafeRelease(m_pFaceFrameReaders[i]);
	}

	// close the Kinect Sensor
	if (pSensor)
	{
		pSensor->Close();
	}

	SafeRelease(pSensor);
}

HRESULT KinectSensorMultiSource::Open() {

	HRESULT hr;

	// connect to sensor
	hr = GetDefaultKinectSensor(&pSensor);

	// open sensor
	if (SUCCEEDED(hr))
	{
		hr = pSensor->Open();
	}

	// open reader
	if (SUCCEEDED(hr))
	{
		hr = pSensor->OpenMultiSourceFrameReader(
			FrameSourceTypes::FrameSourceTypes_Depth |
			FrameSourceTypes::FrameSourceTypes_Color |
			FrameSourceTypes::FrameSourceTypes_Infrared |
			FrameSourceTypes::FrameSourceTypes_Body |
			FrameSourceTypes::FrameSourceTypes_BodyIndex,
			&pSourceReader);
	}

	// define the face frame features
	static const DWORD c_FaceFrameFeatures =
		FaceFrameFeatures::FaceFrameFeatures_BoundingBoxInColorSpace
		| FaceFrameFeatures::FaceFrameFeatures_PointsInColorSpace
		| FaceFrameFeatures::FaceFrameFeatures_RotationOrientation
		| FaceFrameFeatures::FaceFrameFeatures_Happy
		| FaceFrameFeatures::FaceFrameFeatures_RightEyeClosed
		| FaceFrameFeatures::FaceFrameFeatures_LeftEyeClosed
		| FaceFrameFeatures::FaceFrameFeatures_MouthOpen
		| FaceFrameFeatures::FaceFrameFeatures_MouthMoved
		| FaceFrameFeatures::FaceFrameFeatures_LookingAway
		| FaceFrameFeatures::FaceFrameFeatures_Glasses
		| FaceFrameFeatures::FaceFrameFeatures_FaceEngagement;

	if (SUCCEEDED(hr))
	{
		// create a face frame source + reader to track each body in the fov
		for (int i = 0; i < NR_USERS; i++)
		{
			if (SUCCEEDED(hr))
			{
				// create the face frame source by specifying the required face frame features
				hr = CreateFaceFrameSource(pSensor, 0, c_FaceFrameFeatures, &m_pFaceFrameSources[i]);
			}
			if (SUCCEEDED(hr))
			{
				// open the corresponding reader
				hr = m_pFaceFrameSources[i]->OpenReader(&m_pFaceFrameReaders[i]);
			}
		}
	}

	long start = time(0) * 1000;
	long timeout = 2000;	// in seconds
	long timeLeft = timeout;

	if (SUCCEEDED(hr))
	{
		IMultiSourceFrame* p_multisource_frame = NULL;
		std::cout << "Looking for sensor";
		// try to connect to camera
		do {
			timeLeft = timeout - (time(0) * 1000 - start);
			hr = pSourceReader->AcquireLatestFrame(&p_multisource_frame);
			SafeRelease(p_multisource_frame);
			if (SUCCEEDED(hr))
			{
				break;
			}
			// wait
			Sleep(200);
			std::cout << ".";
		} while (!SUCCEEDED(hr) && (timeLeft > 0));

		std::cout << "\n";
		if (!SUCCEEDED(hr))
		{
			std::cout << "Camera is NOT connected...\n";
		}
		else
		{
			std::cout << "Camera found...\n";
		}
	}

	return hr;
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
	IFaceFrame* pFaceFrame[NR_USERS] = {NULL};

	// get multiframe
	hr = pSourceReader->AcquireLatestFrame(&p_multisource_frame);

#ifdef _DEBUG
	if (hr == E_PENDING)
	{
		std::cout << "Frame pending...\n";
	}
#endif

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

		// face - separate reader
		if (SUCCEEDED(hr))
		{
			// iterate through each face reader
			for (int iFace = 0; iFace < NR_USERS; ++iFace)
			{
				// retrieve the latest face frame from this reader
				hr = m_pFaceFrameReaders[iFace]->AcquireLatestFrame(&pFaceFrame[iFace]);
			}
		}

	}	// acquire multisource frame


	// process frames if everything went smoothly
	if (SUCCEEDED(hr)) {
		mSensorMutex.lock();	// class internal data manipulations
		// color
		ProcessColorFrame(p_color_frame, ColorImageStreamHeight, ColorImageStreamWidth, pColorImageBuffer, mColorImageBufferLen);
		// depth
		ProcessDepthFrame(p_depth_frame, DepthStreamHeight, DepthStreamWidth, pDepthBuffer, mDepthBufferLen);
		
		// body - get and refresh pp bodies
		hr = ProcessBodyFrame(p_body_frame);

		// face frame - process after body
		if (SUCCEEDED(hr)) {
			ProcessFaces(pFaceFrame);
		}
		
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

	// release faces
	for (int iFace = 0; iFace < NR_USERS; ++iFace)
	{
		SafeRelease(pFaceFrame[iFace]);
	}

	// return status
	return hr;
}

// --------------------- PROCESSING FUNCTIONS


HRESULT KinectSensorMultiSource::ProcessFaces(IFaceFrame* face_frames[NR_USERS])
{
	
	HRESULT hr = E_FAIL;
	IFaceFrame* pFaceFrame = NULL;

	// iterate through each face reader
	for (int iFace = 0; iFace < NR_USERS; ++iFace)
	{
		pFaceFrame = face_frames[iFace];

		BOOLEAN bFaceTracked = false;
		if (SUCCEEDED(hr) && nullptr != pFaceFrame)
		{
			// check if a valid face is tracked in this face frame
			hr = pFaceFrame->get_IsTrackingIdValid(&bFaceTracked);
		}

		if (SUCCEEDED(hr))
		{

			if (bFaceTracked)
			{
				// valid face
				IFaceFrameResult* pFaceFrameResult = nullptr;
				RectI faceBox = { 0 };
				PointF facePoints[FacePointType::FacePointType_Count];
				Vector4 faceRotation;
				DetectionResult faceProperties[FaceProperty::FaceProperty_Count];

				hr = pFaceFrame->get_FaceFrameResult(&pFaceFrameResult);

				// need to verify if pFaceFrameResult contains data before trying to access it
				if (SUCCEEDED(hr) && pFaceFrameResult != nullptr)
				{
					hr = pFaceFrameResult->get_FaceBoundingBoxInColorSpace(&faceBox);

					if (SUCCEEDED(hr))
					{
						hr = pFaceFrameResult->GetFacePointsInColorSpace(FacePointType::FacePointType_Count, facePoints);
					}

					if (SUCCEEDED(hr))
					{
						hr = pFaceFrameResult->get_FaceRotationQuaternion(&faceRotation);
					}

					if (SUCCEEDED(hr))
					{
						hr = pFaceFrameResult->GetFaceProperties(FaceProperty::FaceProperty_Count, faceProperties);
					}

				}
				else
				{
					// face tracking is not valid - attempt to fix the issue
					// a valid body is required to perform this step
						// check if the corresponding body is tracked 
						// if this is true then update the face frame source to track this body
						IBody* pBody = ppBodies[iFace];
						if (pBody != nullptr)
						{
							BOOLEAN bTracked = false;
							hr = pBody->get_IsTracked(&bTracked);

							UINT64 bodyTId;
							if (SUCCEEDED(hr) && bTracked)
							{
								// get the tracking ID of this body
								hr = pBody->get_TrackingId(&bodyTId);
								if (SUCCEEDED(hr))
								{
									// update the face frame source with the tracking ID
									m_pFaceFrameSources[iFace]->put_TrackingId(bodyTId);
								}
							}
						}
				}

				// release face info
				SafeRelease(pFaceFrameResult);
			}	// /bFaceTracked valid face
		}
	}

	return hr;

}


HRESULT KinectSensorMultiSource::ProcessBodyFrame(IBodyFrame *body_frame) {

	HRESULT hr = E_FAIL;

	// readin body buffer
	hr = body_frame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);

	if (SUCCEEDED(hr)) {
		// do something
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

void KinectSensorMultiSource::GetImageCopyRGBA(cv::Mat &dst) {
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

void KinectSensorMultiSource::GetImageCopyRGB(cv::Mat &dst) {
	mSensorMutex.lock();
	cv::Mat cv_img(ColorImageStreamHeight, ColorImageStreamWidth, CV_8UC4, reinterpret_cast<void*>(pColorImageBuffer));
	cv::Mat resized;
	cv::resize(cv_img, resized, cv::Size(mColorWidth, mColorHeight));
	// rgba to rgb
	cv::cvtColor(resized, resized, CV_RGBA2RGB);
	dst = resized;
	mSensorMutex.unlock();
}

void KinectSensorMultiSource::GetImageRGBA(cv::Mat &dst) {
	cv::Mat cv_img(ColorImageStreamHeight, ColorImageStreamWidth, CV_8UC4, reinterpret_cast<void*>(pColorImageBuffer));
	dst = cv_img;
}


void KinectSensorMultiSource::GetImageCopyDepth(cv::Mat &dst) {
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
