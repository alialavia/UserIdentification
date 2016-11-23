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
	pSourceReader(nullptr),
	mColorWidth(1920),
	mColorHeight(1080),
	mDepthImageWidth(480),
	mDepthImageHeight(320),
	pColorImageBuffer(nullptr),
	pDepthBuffer(nullptr),
	pBodyIndexBuffer(nullptr),
	pSensor(nullptr)
{
	// init face frame source and readers
	for (int i = 0; i < NR_USERS; i++)
	{
		m_pFaceFrameSources[i] = nullptr;
		m_pFaceFrameReaders[i] = nullptr;
	}
}

// Careful! Extracted images will be deleted if the sensor streamer runs out of scope
KinectSensorMultiSource::~KinectSensorMultiSource()
{
	Close();

	// cleanup buffers

	// color
	if (pColorImageBuffer != nullptr)
	{
		delete[] pColorImageBuffer;
		pColorImageBuffer = nullptr;
	}
	// depth
	if (pDepthBuffer != nullptr)
	{
		delete[] pDepthBuffer;
		pDepthBuffer = nullptr;
	}
	// body data
	for (int i = 0; i < _countof(ppBodies); ++i)
	{
		SafeRelease(ppBodies[i]);
	}
	// body index
	if (pBodyIndexBuffer != nullptr)
	{
		delete[] pBodyIndexBuffer;
		pBodyIndexBuffer = nullptr;
	}
	// face data
}

void KinectSensorMultiSource::Close()
{
	// release multisource reader
	SafeRelease(pSourceReader);

	// face readers
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

HRESULT KinectSensorMultiSource::Open(int timeout)
{
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
	long timeLeft = timeout;

	if (SUCCEEDED(hr))
	{
		IMultiSourceFrame* p_multisource_frame = nullptr;
		std::cout << "Looking for sensor";
		// try to connect to camera
		do
		{
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
		}
		while (!SUCCEEDED(hr) && (timeLeft > 0));

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

HRESULT KinectSensorMultiSource::AcquireFrame()
{
	if (!pSensor)
	{
		return E_FAIL;
	}

	HRESULT hr = E_PENDING;

	// allocate frames
	IMultiSourceFrame* p_multisource_frame = nullptr;
	IDepthFrame* p_depth_frame = nullptr;
	IColorFrame* p_color_frame = nullptr;
	IInfraredFrame* p_infrared_frame = nullptr;
	IBodyFrame* p_body_frame = nullptr;
	IBodyIndexFrame* p_bodyindex_frame = nullptr;
	IFaceFrame* pFaceFrames[NR_USERS] = {nullptr};

	// get multiframe
	hr = pSourceReader->AcquireLatestFrame(&p_multisource_frame);

#ifdef _DEBUG
	if (hr == E_PENDING)
	{
		//std::cout << "Frame pending...\n";
	}
#endif

	// IF FPS > FPS_kINECT:=30
	if (SUCCEEDED(hr))
	{
		// color
		if (SUCCEEDED(hr))
		{
			IColorFrameReference* cfr = nullptr;

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
			IDepthFrameReference* dfr = nullptr;

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
			IInfraredFrameReference* ifr = nullptr;

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
			IBodyFrameReference* bfr = nullptr;

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
			IBodyIndexFrameReference* bifr = nullptr;

			hr = p_multisource_frame->get_BodyIndexFrameReference(&bifr);
			if (SUCCEEDED(hr))
			{
				hr = bifr->AcquireFrame(&p_bodyindex_frame);
			}
			SafeRelease(bifr);
		}
	} // acquire multisource frame

	// face - separate reader
	if (SUCCEEDED(hr))
	{
		// iterate through each face reader
		for (int iFace = 0; iFace < NR_USERS; ++iFace)
		{
			// retrieve the latest face frame from this reader
			hr = m_pFaceFrameReaders[iFace]->AcquireLatestFrame(&pFaceFrames[iFace]);
		}
	}

	// process frames if everything went smoothly
	if (SUCCEEDED(hr))
	{
		mSensorMutex.lock(); // class internal data manipulations
		// color
		ProcessColorFrame(p_color_frame, ColorImageStreamHeight, ColorImageStreamWidth, pColorImageBuffer, mColorImageBufferLen);
		// depth
		ProcessDepthFrame(p_depth_frame, DepthStreamHeight, DepthStreamWidth, pDepthBuffer, mDepthBufferLen);
		// body index
		ProcessBodyIndexFrame(p_bodyindex_frame, BodyIndexStreamHeight, BodyIndexStreamWidth, pBodyIndexBuffer, mBodyIndexBufferLen);
		// body - get and refresh pp bodies
		hr = ProcessBodyFrame(p_body_frame);

		// face frame - process after body
		if (SUCCEEDED(hr))
		{
			hr = ProcessFaceFrames(pFaceFrames);
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
		SafeRelease(pFaceFrames[iFace]);
	}

	// return status
	return hr;
}

// --------------------- PROCESSING FUNCTIONS

HRESULT KinectSensorMultiSource::ProcessBodyIndexFrame(IBodyIndexFrame* index_frame, int& height, int& width, BYTE* & buffer, UINT& buffer_len)
{
	IFrameDescription* frameDesc = nullptr;
	HRESULT hr = E_FAIL;

	// current image buffer length
	UINT nBufferLen;

	hr = index_frame->get_FrameDescription(&frameDesc);

	// readin image buffer
	if (SUCCEEDED(hr))
	{
		// get stream height and width
		if (
			SUCCEEDED(frameDesc->get_Height(&height)) &&
			SUCCEEDED(frameDesc->get_Width(&width))
		)
		{
			// allocate buffer
			nBufferLen = height * width * sizeof(BYTE);

			// allocate more memory if necessary
			if (nBufferLen > buffer_len)
			{
				if (buffer != nullptr)
				{
					delete[] buffer;
				}
				buffer = new BYTE[nBufferLen];
				buffer_len = nBufferLen;
			}

			hr = index_frame->CopyFrameDataToArray(nBufferLen, reinterpret_cast<BYTE*>(buffer));
		}
	}

	SafeRelease(frameDesc);
	// DO NOT RELEASE FRAME HERE! THIS IS A PURE PROCESSING FUNCTION

	return hr;
}

HRESULT KinectSensorMultiSource::ProcessFaceFrames(IFaceFrame* face_frames[NR_USERS])
{
	HRESULT hr = E_FAIL;
	IFaceFrame* pFaceFrame = nullptr;

	// iterate through each face reader
	for (int iFace = 0; iFace < NR_USERS; ++iFace)
	{
		pFaceFrame = face_frames[iFace];
		
		FaceData new_face;
		BOOLEAN bFaceTracked = false;

		if (nullptr != pFaceFrame)
		{
			// check if a valid face is tracked in this face frame
			hr = pFaceFrame->get_IsTrackingIdValid(&bFaceTracked);
		}

		if (SUCCEEDED(hr))
		{
			// valid face
			if (bFaceTracked)
			{
				// TODO: check why face tracking is not working
				IFaceFrameResult* pFaceFrameResult = nullptr;
				hr = pFaceFrame->get_FaceFrameResult(&pFaceFrameResult);

				// need to verify if pFaceFrameResult contains data before trying to access it
				if (SUCCEEDED(hr) && pFaceFrameResult != nullptr)
				{
					// everything is fine - extract data from face frame result
					hr = pFaceFrameResult->get_FaceBoundingBoxInColorSpace(&new_face.boundingBox);

					if (SUCCEEDED(hr))
					{
						hr = pFaceFrameResult->get_FaceBoundingBoxInInfraredSpace(&new_face.boundingBoxIR);
					}
					if (SUCCEEDED(hr))
					{
						hr = pFaceFrameResult->GetFacePointsInColorSpace(FacePointType::FacePointType_Count, new_face.Points);
					}
					if (SUCCEEDED(hr))
					{
						hr = pFaceFrameResult->GetFacePointsInInfraredSpace(FacePointType::FacePointType_Count, new_face.PointsIR);
					}
					if (SUCCEEDED(hr))
					{
						hr = pFaceFrameResult->get_FaceRotationQuaternion(&new_face.Rotation);
					}
					if (SUCCEEDED(hr))
					{
						hr = pFaceFrameResult->GetFaceProperties(FaceProperty::FaceProperty_Count, new_face.Properties);
					}

					// set tracking status
					if (SUCCEEDED(hr)){
						new_face.tracked = true;
					}
					
				}
				SafeRelease(pFaceFrameResult);
			}
			else
			{
				// update tracking id from body
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
		}

		// save
		mFaces[iFace] = new_face;

		// DO NOT RELEASE FRAME HERE! THIS IS A PURE PROCESSING FUNCTION

	}	// end for

	return hr;
}

HRESULT KinectSensorMultiSource::ProcessBodyFrame(IBodyFrame* body_frame)
{
	HRESULT hr = E_FAIL;

	// readin body buffer
	hr = body_frame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);

	if (SUCCEEDED(hr))
	{
		// do something
	}

	return hr;
}

HRESULT KinectSensorMultiSource::ProcessColorFrame(IColorFrame* color_frame, int& height, int& width, RGBQUAD* & buffer, UINT& buffer_len) const
{
	IFrameDescription* frameDesc = nullptr;
	HRESULT hr = E_FAIL;

	// output parameters
	ColorImageFormat targetFormat = ColorImageFormat_Bgra;

	// current information
	ColorImageFormat imageFormat;

	// current image buffer length
	UINT nBufferLen;

	hr = color_frame->get_FrameDescription(&frameDesc);

	// readin image buffer
	if (SUCCEEDED(hr))
	{
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
			if (nBufferLen > buffer_len)
			{
				if (buffer != nullptr)
				{
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

// The pixel values in this frame are 8 - bit unsigned integers, where 0 - 5 map directly to the BodyData index in the BodyFrame.
// Values greater than the value obtained from BodyCount indicate the pixel is part of the background, not associated with a tracked body.
HRESULT KinectSensorMultiSource::ProcessDepthFrame(IDepthFrame* depth_frame, int& height, int& width, UINT16* & buffer, UINT& buffer_len) const
{
	HRESULT hr;

	IFrameDescription* frameDesc = nullptr;
	USHORT nDepthMinReliableDistance = 0;
	USHORT nDepthMaxDistance = 0;

	// current image buffer length
	UINT nBufferLen;

	UINT nBufferSize = 0;

	hr = depth_frame->get_FrameDescription(&frameDesc);

	// readin image buffer
	if (SUCCEEDED(hr))
	{
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
			if (nBufferLen > buffer_len)
			{
				if (buffer != nullptr)
				{
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

// --------------------- DEEP COPY DATA ACCESS

void KinectSensorMultiSource::GetImageCopyBodyIndexColored(cv::Mat& dst) const
{
	mSensorMutex.lock();

	// convert to mat
	// uint 8bit (0-256), players: 0-5
	cv::Mat cv_img(BodyIndexStreamHeight, BodyIndexStreamWidth, CV_8UC1, reinterpret_cast<void*>(pBodyIndexBuffer));

	cv::Mat resized;
	cv::resize(cv_img, resized, cv::Size(BodyIndexStreamWidth, BodyIndexStreamHeight));
	// alternatively, if dimension stays the same: dst = cv_img.copyTo(dst) - copies everything, not only header (in contrast to .clone())

	cv::Vec3b user_colors[6];
	user_colors[0] = cv::Vec3b(255, 0, 0);
	user_colors[1] = cv::Vec3b(0, 255, 0);
	user_colors[2] = cv::Vec3b(0, 0, 255);
	user_colors[3] = cv::Vec3b(255, 0, 255);
	user_colors[4] = cv::Vec3b(255, 255, 0);
	user_colors[5] = cv::Vec3b(0, 255, 255);

	// convert color
	cv::Mat colored;
	cv::cvtColor(resized, colored, cv::COLOR_GRAY2BGR);

	int user_id;
	for (int i = 0; i < resized.cols; i++)
	{
		for (int j = 0; j < resized.rows; j++)
		{
			user_id = resized.at<uchar>(j, i);
			if (user_id < NR_USERS)
			{
				colored.at<cv::Vec3b>(j, i) = user_colors[user_id];
			}
		}
	}

	// set pointer of dst to resized, colored image
	dst = colored;
	mSensorMutex.unlock();
}

void KinectSensorMultiSource::GetImageCopyRGBA(cv::Mat& dst) const
{
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

void KinectSensorMultiSource::GetImageCopyRGB(cv::Mat& dst) const
{
	mSensorMutex.lock();
	cv::Mat cv_img(ColorImageStreamHeight, ColorImageStreamWidth, CV_8UC4, reinterpret_cast<void*>(pColorImageBuffer));
	cv::Mat resized;
	cv::resize(cv_img, resized, cv::Size(mColorWidth, mColorHeight));
	// rgba to rgb
	cv::cvtColor(resized, resized, CV_RGBA2RGB);
	dst = resized;
	mSensorMutex.unlock();
}

void KinectSensorMultiSource::GetImageCopyRGBSkeleton(cv::Mat& dst) const
{
	mSensorMutex.lock();
	cv::Mat cv_img(ColorImageStreamHeight, ColorImageStreamWidth, CV_8UC4, reinterpret_cast<void*>(pColorImageBuffer));
	cv::Mat resized;
	cv::resize(cv_img, resized, cv::Size(mColorWidth, mColorHeight));
	// rgba to rgb
	cv::cvtColor(resized, resized, CV_RGBA2RGB);
	dst = resized;
	mSensorMutex.unlock();
}

void KinectSensorMultiSource::GetImageCopyDepth(cv::Mat& dst) const
{
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

// --------------------- DIRECT DATA ACCESS

void KinectSensorMultiSource::GetImageRGBA(cv::Mat& dst)
{
	cv::Mat cv_img(ColorImageStreamHeight, ColorImageStreamWidth, CV_8UC4, reinterpret_cast<void*>(pColorImageBuffer));
	dst = cv_img;
}

 HRESULT KinectSensorMultiSource::GetSensorReference(IKinectSensor* &s)
{
	if(pSensor == nullptr)
	{
		return E_FAIL;
	}
	s = pSensor;
	return S_OK;
}

IBody** KinectSensorMultiSource::GetBodyDataReference()
{
	return ppBodies;
}

FaceData* KinectSensorMultiSource::GetFaceDataReference()
{
	return mFaces;
}
