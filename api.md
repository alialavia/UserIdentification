# UIDS - API



## User Identification



#### User Manager API
*Using the user manager is used allows easy access to server side identification information*

The following methods are provided (for detailed usage please refer to the source code):
- `GetUserandTrackingID` Get the fixed ID and the corresponding Kinect tracking ID for all users in the scene
- `GetSceneProfilePictures` Get the profile pictures for all users in the scene
- `GetAllProfilePictures` Get the profile pictures of all users in the database
- `GetUserID` Get the fixed user ID from a picture of a persons face

#### Direct Request Generation
*Besides using the UserManager, requests can be directly generated and sent to the server.*
Please note that:
- Depending on the server version, not all request types are supported. Check out the individual `config.py` files for the servers under `uids/services`
- All client side available requests and responses are listed in `include/io/RequestTypes.h` respectively in `include/io/ResponseTypes.h`
- For further examples, see: `applications/online_emu` or `test/v2`

**Example Usage**

```cpp
std::vector<cv::Mat> face_snapshots;
...
// extract faces here
...
// config to server connection
io::TCPClient server_conn;
server_conn.Config("127.0.0.1", 8080);
// connect to server
server_conn.Connect();
// allocate request handler
io::NetworkRequestHandler request_handler;
// generate request
io::ImageIdentification *id_request = new io::ImageIdentification(&server_conn, face_snapshots);
request_handler.addRequest(id_request);
// process request(s)
request_handler.processAllPendingRequests();
// handle different response types
io::IdentificationResponse response;
if (request_handler->PopResponse(&response))
{
    std::cout << "--- Identification\n";
}
io::OKResponse ok_response;
if (request_handler->PopResponse(&ok_response, request_lookup))
{
    std::cout << "--- OK\n";
}
io::ErrorResponse err_response;
if (request_handler->PopResponse(&err_response, request_lookup))
{
    std::cout << "--- Error\n";
}
// close connection to server
server_conn.Close();

```


## Sensor Interface and Tracking Interface

#### io::KinectSensorMultiSource
*Allows fast and flexible access to all Kinect sensor streams*

**Sensor connection:**
- `HRESULT Open(int timeout = 5000)` Connect to Kinect sensor stream
- `void Close()` Terminate connection to Kinect

**Stream access:**
- `GetImageCopyRGBA` Extract RGBA image
- `etImageCopyRGB` Extract RGB image
- `GetImageCopyRGBSubtracted` Extract RGB image with subtracted background
- `GetImageCopyDepth` Extract depth image
- `GetImageCopyDepth8UThresholded` Extract thresholded depth image (for visualization)
- `GetImageCopyBodyIndex` Get a body index map
- `GetImageCopyBodyIndexColored` Get a body index visualization
- `GetImageCopyRGBSkeleton` Extract RGB image with marked skeleton joints

**Raw data access**
- `HRESULT GetSensorReference(IKinectSensor* &s)` Get a reference to the sensor
- `IBody** GetBodyDataReference()` Get a reference to the current body data
- `FaceData* GetFaceDataReference()` Get a reference to the current face data


**Example Usage**

```cpp
// init sensor stream
io::KinectSensorMultiSource k;
HRESULT hr;
cv::Mat image;

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
		k.GetImageCopyRGB(image);
		// perform your task here...
    }
 }
```

#### tracking::SkeletonTracker
*Allows to extract skeleton tracking information from a Kinect sensor source*

**Initialization**
- `SkeletonTracker(IKinectSensor* sensor)` Create the tracker
- `HRESULT Init()` Connect to Kinect sensor stream

**Skeleton Joint Extraction**
- `ExtractJoints` Extract the body joint objects from the kinect source
- `GetJointPosition` Extract body joint coordinates
- `GetJointProjections` Extract the 2D coordinates of body joints projected onto the RGB or the depth space
- `GetFaceBoundingBoxesRobust` Get the face bounding boxes
- `GetUserSceneIDs` Get the current user tracking IDs

**Visualization**
- `RenderFaceBoundingBoxes` Render face bounding boxes onto an image
- `RenderAllBodyJoints` Render skeleton joints onto an image


**Example Usage**

```cpp
// init sensor stream
io::KinectSensorMultiSource k;
HRESULT hr;
cv::Mat image;

// initialize sensor
if (FAILED(k.Open())) {
    std::cout << "Initialization failed" << std::endl;
    return -1;
}

// get sensor reference
IKinectSensor* pSensor = nullptr;
if (FAILED(k.GetSensorReference(pSensor)))
{
    std::cout << "Sensor is not initialized" << std::endl;
    return -1;
}

// skeleton tracker
tracking::SkeletonTracker st(pSensor);
if (FAILED(st.Init()))
{
    std::cout << "Skeleton tracker initialization failed" << std::endl;
    return -1;
}

// allocate data
std::vector<cv::Rect2f> bounding_boxes;
std::vector<cv::Point3f> user_positions;
std::vector<int> user_scene_ids;

while (true)
{
    // polling
    hr = k.AcquireFrame();
    // check if there is a new frame available
    if (SUCCEEDED(hr)) {
        // extract raw skeleton data
        IBody** bodies = k.GetBodyDataReference();
        st.ExtractJoints(bodies);
        // extract face bounding boxes from skeletons and corresponding users ids
        st.GetFaceBoundingBoxesRobust(bounding_boxes, user_scene_ids, base::ImageSpace_Color);
        st.GetJointPosition(base::JointType_SpineMid, user_positions);
    }
}
```


#### tracking::FaceTracker
*Allows to extract facial tracking data from a Kinect sensor source *

**Initialization**
- `FaceTracker(IKinectSensor* sensor)` Create the tracker

**Face Tracking**
- `ExtractFacialData` Extract the raw face tracking data from a Kinect sensor source
- `GetUserSceneIDs` Get the current tracking IDs
- `GetFaceBoundingBoxesRobust` Get the face bounding boxes
- `GetFaces` Get the face tracking data

**Visualization**
- `RenderFaceBoundingBoxes` Render face bounding boxes ont an image
- `RenderFaceFeatures` Render facial landmarks onto an image


**Example Usage**

```cpp
// init sensor stream
io::KinectSensorMultiSource k;
HRESULT hr;
cv::Mat image;

// initialize sensor
if (FAILED(k.Open())) {
    std::cout << "Initialization failed" << std::endl;
    return -1;
}
// get sensor reference
IKinectSensor* pSensor = nullptr;
if (FAILED(k.GetSensorReference(pSensor)))
{
    std::cout << "Sensor is not initialized" << std::endl;
    return -1;
}

// initialize face tracker
tracking::FaceTracker ft(pSensor);

// allocate data
std::vector<tracking::Face> faces;
std::vector<int> user_scene_ids;

while (true)
{
    // polling
    hr = k.AcquireFrame();
    // check if there is a new frame available
    if (SUCCEEDED(hr)) {
        // extract raw face data
        FaceData* face_data_raw = k.GetFaceDataReference();
        ft.ExtractFacialData(face_data_raw);
        // get faces
        ft.GetFaces(faces, user_scene_ids);
    }
}
```

