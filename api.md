# UIDS - API



## Sensor Interface and Utility Modules

#### io::KinectSensorMultiSource
*Allows fast and flexible access to all Kinect sensor streams.*



**Sensor connection:**
- `HRESULT Open(int timeout = 5000)` Connect to Kinect sensor stream
- `void Close()` Terminate connection to Kinect

**Stream access:**
- `void GetImageCopyRGBA(cv::Mat& dst) const` Extract RGBA image
- `void GetImageCopyRGB(cv::Mat& dst) const` Extract RGB image
- `void GetImageCopyRGBSubtracted(cv::Mat& dst) const` Extract RGB image with subtracted background
- `void GetImageCopyDepth(cv::Mat& dst) const` Extract depth image
- `void GetImageCopyDepth8UThresholded(cv::Mat& dst) const` Extract thresholded depth image (for visualization)
- `void GetImageCopyBodyIndex(cv::Mat& dst) const` Get a body index map
- `void GetImageCopyBodyIndexColored(cv::Mat& dst) const` Get a body index visualization
- `void GetImageCopyRGBSkeleton(cv::Mat& dst) const` Extract RGB image with marked skeleton joints

**Raw data access**
- `HRESULT GetSensorReference(IKinectSensor* &s)` Get a reference to the sensor
- `IBody** GetBodyDataReference()` Get a reference to the current body data
- `FaceData* GetFaceDataReference()` Get a reference to the current face data


**Example Usage**

```cpp
// init sensor stream
io::KinectSensorMultiSource k;
HRESULT hr;
cv::Mat color_image;

// initialize sensor
if (FAILED(k.Open())) {
    std::cout << "Initialization failed" << std::endl;
    return -1;
}

while (true)
{
    // polling
    hr = k.AcquireFrame();
    // check if there is a new frame available
    if (SUCCEEDED(hr)) {
    	// get color image
		k.GetImageCopyRGB(color_image);
		// perform your task here...
    }
 }

```

#### ```io::KinectSensorMultiSource ```

-----

#### Examples


```cpp
// init sensor stream
io::KinectSensorMultiSource sensor;

while (true)
{
    // polling
    hr = k.AcquireFrame();

    // check if there is a new frame available
    if (SUCCEEDED(hr)) {
    	// get color image
		k.GetImageCopyRGB(color_image);

    }
 }

```

## User Identification




io::KinectSensorMultiSource k;




