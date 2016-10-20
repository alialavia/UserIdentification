
#include <string>
#include <iostream>
#include <base/UserIdentification.h>

// pointer safe release
#include <strsafe.h>


// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// forward declarations
struct IColorFrameSource;
struct IDepthFrameSource;
struct IColorFrameReader;
struct IDepthFrameReader;
struct IFrameDescription;
struct IKinectSensor;


// Kinect SDK 2
#include <Kinect.h>



// --------------------------------

// forward declaration
class KinectSensor;

template 
<
	typename SensorType, 
	typename SourceType,
	typename ReaderType, 
	typename DescriptionType,
	typename DataType,
	typename DataTypeRaw
>
class SensorStream {
public:
	// set sensor
	SensorStream(SensorType * Sensor) :pSensor(Sensor)
	{
	}

	// cleanup
	~SensorStream() {
		SafeRelease(pSource);
		SafeRelease(pReader);
		SafeRelease(pDescription);
	}

	// update method: implement in specific stream type
	virtual void Update() = 0;
	// process method: implement in specific stream type - transforms pDataRaw to pDataData
	virtual void ProcessData() = 0;

	SensorType* pSensor;			// pointer to sensor:				 set in sensor stream class
	SourceType* pSource;			// pointer to stream source:		 set in sensor stream class
	ReaderType* pReader;			// pointer to stream reader:		 set in sensor stream class
	DescriptionType* pDescription;	// pointer to description container: set in sensor stream class

	// data container
	DataType* pData; // is set in specific stream type (invoced custom processing function)
	// raw data container
	DataTypeRaw* pDataRaw; // is set in sensor class stream type

	// access handle: multi threading
	WAITABLE_HANDLE handle;
	// infraredFrameReader->SubscribeFrameArrived(&handle);
	// waitForHandle()
};


// kinect sensor stream interface
/*

call order in loop:
Update();
which calls:
ProcessData();

*/
template <
	typename SourceType,
	typename ReaderType, 
	typename DescriptionType,
	typename DataType,
	typename DataTypeRaw
>
class KinectSensorStream : public SensorStream <IKinectSensor, SourceType, ReaderType, DescriptionType, DataType, DataTypeRaw> {	// set sensor type
public:
	KinectSensorStream(IKinectSensor * InputSensor) : SensorStream<IKinectSensor, SourceType, ReaderType, DescriptionType, DataType, DataTypeRaw>(InputSensor) {	// pass sensor pointer

	};

	~KinectSensorStream() {
		// kinect specific stream cleanup
	}


	// transforms raw data to data type
	virtual void ProcessData() = 0;

	// call directly on stream to update data
	void Update() {
		if (!pReader) {
			return;
		}

		// reset data and description
		DataTypeRaw* pDataRaw = NULL;
		DataType* pData = NULL;
		IFrameDescription* pFrameDescription = NULL;

		// read data - same for every sensor stream (also bodyframe stream)
		HRESULT hr = pReader->AcquireLatestFrame(&pDataRaw);

		if (SUCCEEDED(hr))
		{
			INT64 nTime = 0;
			hr = pDataRaw->get_RelativeTime(&nTime);	// needed to calculate frame rate in ProcessData 

			// get description
			if (SUCCEEDED(hr))
			{
				// get_FrameDescription - must be present for all raw stream data types
				hr = pDataRaw->get_FrameDescription(&pFrameDescription);
			}

			// process raw data (get readable data)
			if (SUCCEEDED(hr))
			{
				ProcessData();
			}

		}

		// remember: release data frame
		// If the data frame you point to has data, it will return E_PENDING until it's cleared
		SafeRelease(pDescription);
		SafeRelease(pDataRaw);
	}

	// called in constructor of every stream type - initializes the stream
	template <typename FUNC, typename FUNC2, typename FUNC3>
	bool InitStream(FUNC getSource_Functor, FUNC2 openReader_Functor, FUNC3 getDescription_Functor)
	{
		HRESULT hResult;

		// init stream source: functor called from sensor on source
		hResult = (pSensor->*getSource_Functor)(&pSource);
		if (FAILED(hResult)) {
			std::cout << typeid(getSource_Functor).name() << " failed." << '\n';
			return false;
		}

		// init stream source reader: functor called from source on reader
		hResult = (pSource->*openReader_Functor)(&pReader);
		if (FAILED(hResult)) {
			std::cout << typeid(openReader_Functor).name() << " failed." << '\n';
			return false;
		}

		// get source description
		hResult = (pSource->*getDescription_Functor)(&pDescription);
		if (FAILED(hResult)) {
			std::cout << typeid(getDescription_Functor).name() << " failed." << '\n';
			return false;
		}

		return true;
	}

	// TODO: is this needed?
	//template <typename OBJECT, typename FUNC>
	//bool InitStreamWithRef(OBJECT obj, FUNC getSource_Functor)
	//{
	//	// pointer to member of object pointer
	//	(obj->*getSource_Functor)(&pSource);

	//	return true;
	//}

	// start the stream


};



// ---------------------------------------- IMPLEMENT SENSOR STREAMS


// color
/*
Call order
ColorStream* cs = new ColorStream(pSensor);
cs.ConfigureStream(1920, 1080);
*/
class ColorStream : public KinectSensorStream<IColorFrameSource, IColorFrameReader, IFrameDescription, cv::Mat, IColorFrame> {	// init kinect sensor with source and reader
public:

	const int mWidth;
	const int mHeight;
	const ColorImageFormat mColorFormat;

	ColorStream(IKinectSensor * InputSensor, int imgWidth = 1920, int imgHeight = 1080, ColorImageFormat ImageFormat = ColorImageFormat_Bgra) :
		KinectSensorStream<IColorFrameSource, IColorFrameReader, IFrameDescription, cv::Mat, IColorFrame>(InputSensor),
		mWidth(imgWidth), mHeight(imgHeight), mColorFormat(ImageFormat){	// create kinect sensor
		// stream specific initialization
		InitStream(&IKinectSensor::get_ColorFrameSource, &IColorFrameSource::OpenReader, &IColorFrameSource::get_FrameDescription);
	}

	// main function implementation
	void ProcessData() {
		// description loaded at this point
		HRESULT hr = E_FAIL;
		int nWidth = 0;
		int nHeight = 0;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		UINT nBufferSize = 0;
		RGBQUAD *pBuffer = nullptr;

		// create heap storage for color pixel data in RGBX format
		RGBQUAD* pColorRGBX;
		pColorRGBX = new RGBQUAD[mWidth * mHeight];
	
		if (SUCCEEDED(pDescription->get_Width(&nWidth))
			&& SUCCEEDED(pDescription->get_Height(&nHeight))
			&& SUCCEEDED(pDataRaw->get_RawColorImageFormat(&imageFormat))
			)
		{
			if (imageFormat == mColorFormat)
			{
				hr = pDataRaw->AccessRawUnderlyingBuffer(&nBufferSize, reinterpret_cast<BYTE**>(&pBuffer));
			}
			else if (pColorRGBX)
			{
				pBuffer = pColorRGBX;
				nBufferSize = mWidth * mHeight * sizeof(RGBQUAD);
				hr = pDataRaw->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(pBuffer), mColorFormat);
			}
			else
			{
				hr = E_FAIL;
			}
		}

		// convert
		if (SUCCEEDED(hr))
		{
			cv::Mat img1(nHeight, nWidth, CV_8UC4,reinterpret_cast<void*>(pBuffer));
			cv::imshow("Color Only", img1);
		}
		else {
			// no image
		}

		// cleanup
		if (pColorRGBX)
		{
			delete[] pColorRGBX;
			pColorRGBX = NULL;
		}
		if (pBuffer)
		{
			delete[] pBuffer;
			pBuffer = NULL;
		}
	}
};

/*
// depth
class DepthStream : public KinectSensorStream<IDepthFrameSource, IDepthFrameReader, IFrameDescription, cv::Mat, IDepthFrame> {	// init kinect sensor with source and reader
public:
	DepthStream(IKinectSensor * InputSensor) : KinectSensorStream<IDepthFrameSource, IDepthFrameReader, IFrameDescription, cv::Mat, IDepthFrame>(InputSensor) {	// create kinect sensor
		InitStream(&IKinectSensor::get_DepthFrameSource, &IDepthFrameSource::OpenReader, &IDepthFrameSource::get_FrameDescription);
	}
	void ProcessData() {

	}
};

// infrared
class InfraredStream : public KinectSensorStream<IInfraredFrameSource, IInfraredFrameReader, IFrameDescription, cv::Mat, IInfraredFrame> {	// init kinect sensor with source and reader
public:
	InfraredStream(IKinectSensor * InputSensor) : KinectSensorStream<IInfraredFrameSource, IInfraredFrameReader, IFrameDescription, cv::Mat, IInfraredFrame>(InputSensor) {	// create kinect sensor
		InitStream(&IKinectSensor::get_InfraredFrameSource, &IInfraredFrameSource::OpenReader, &IInfraredFrameSource::get_FrameDescription);
	}
	void ProcessData() {

	}
};

*/

/*
// body frame
// specifier: IBodyFrame
// TODO: no Description - implement custom descrition class which provides body count

// Custom body frame description: unified interface
struct IBodyFrameDescription {

};

// unified stream interface
struct IBodyFrameSourceExtended : public IBodyFrameSource {

	// custom body frame description constructor
	HRESULT get_FrameDescription(IBodyFrameDescription *pDescription) {

	}
};

// processed body data type
class BodyDescriptor {

};

class BodyStream : public KinectSensorStream<IBodyFrameSourceExtended, IInfraredFrameReader, IBodyFrameDescription, BodyDescriptor, IBodyFrame> {	// init kinect sensor with source and reader
public:
	BodyStream(IKinectSensor * InputSensor) : KinectSensorStream<IBodyFrameSourceExtended, IInfraredFrameReader, IBodyFrameDescription, BodyDescriptor, IBodyFrame>(InputSensor) {	// create kinect sensor
		InitStream(&IKinectSensor::get_InfraredFrameSource, &IBodyFrameSourceExtended::OpenReader, &IBodyFrameSourceExtended::get_FrameDescription);
	}
	void ProcessData() {

	}
};
*/

// ---------------------------------------- KINECT SENSOR CLASS

class KinectSensor {
public:
	KinectSensor() :
		pColorStream(nullptr)
	{

	}

	~KinectSensor() {

		// shut down streams
		delete pColorStream;

		// close the Kinect Sensor
		if (pSensor)
		{
			pSensor->Close();
		}

		SafeRelease(pSensor);
	}

	// Init sensor
	bool Init() {
		HRESULT hResult;
		// connect to sensor
		hResult = GetDefaultKinectSensor(&pSensor);

		if (FAILED(hResult)) {
			std::cout << "Error : GetDefaultKinectSensor" << std::endl;
			return false;
		}

		// open sensor
		hResult = pSensor->Open();
		if (FAILED(hResult)) {
			std::cout << "Error : IKinectSensor::Open()" << std::endl;
			return false;
		}

		return true;
	}

	// update all sensor streams
	void Update() {

		// color
		if (pColorStream != nullptr) {
			pColorStream->Update();
		}
		// depth

	}

	void RegisterDepthStream() {
		if (!pSensor)
			return;
	}

	void RegisterBodyStream() {
		if (!pSensor)
			return;
	}

	void RegisterColorStream() {
		if (!pSensor)
			return;
		// bind color stream
		pColorStream = new ColorStream(pSensor, 800, 600);
	}

	private:

		IKinectSensor* pSensor;
		ColorStream* pColorStream;
		// depth stream 
		// ..

};