# User Identification

- Automated Build of the identification node on DockerHub: https://hub.docker.com/r/matbloch/user_identification_node/

## System Requirements

- Windows 8/10 x64 (for Kinect SDK 2 support)
- USB3 controller compatible with Kinect (see [Confirmed List of USB 3.0 Controllers for Kinect v2](https://social.msdn.microsoft.com/Forums/en-US/bb379e8b-4258-40d6-92e4-56dd95d7b0bb/confirmed-list-of-usb-30-pcie-cardslaptopsconfigurations-which-work-for-kinect-v2-during?forum=kinectv2sdk))


## Setup Guide

### 1. Building the Library and Client
**Note**: This guide for Windows 8/10. Please install the components in the ++mentioned order++.

#### Dependencies

1.1 **Kinect for Windows SDK 2.x**
- Download and install Kinect for Windows SDK 2.0 (installer)
- Add `C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\bin` to you Systems PATH variable
- Restart

1.2 **CMake**
- Download the latest Cmake installer tool from cmake.org
- Choose add CMake to Path variable
- Test in Console cmake --version

1.3 **OpenCV 3.x**
- Go to [opencv.org](http://opencv.org) and download the precompiled binaries (Visual Studio V12/14 is supported, otherwise you have to build the libraries for VS yourself)
- Set environmentvariable `OPENCV_DIR` (so CMake is able to find library)
Example for installation path in: `C:\lib\OpenCV31\build`
```bash
setx -m OPENCV_DIR "C:\lib\OpenCV31\build"
```
- Add binary path to `PATH` variable (so that DLLs can be included at runtime)
Example for: Visual Studio 14 x64 binaries installed under `C:\lib\opencv31\build`
```bash
set PATH=%PATH%;C:\lib\opencv31\build\x64\vc14\bin;
```

---------------

#### Building from Source with Microsoft Visual Studio

1.4 **Microsoft Visual Studio**
- Go to [visualstudio.com](https://www.visualstudio.com/), download and install MVS 12 or higher

1.5 **Visual C++ Compiler**
- Start Microsoft Visual Studio
- Create new project > Visual C++ and select "Install Visual C++ 2015 Tools for Windows Desktop"

1.6 **Build the project using CMAKE**
1. Download and install CMake
2. Generate Visual Studio project:
Open command prompt and execute:
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
```
Visual Studio is the default generator under Windows but x64 version maybe needs to be specified.
3. Open the generated project in the `build` folder using MVS
4. Compile the project in MVS (e.g. by selecting components or using the predefined targets for a full installation `CMakePredefinedTargets > ALL_BUILD`)


### 2. Server Installation

See manual in the subfolder `uids`.

## Running the User Identification System

1. Connect the Kinect Camera to your Computer through a USB3 Port
2. Start the identification server using **With Boot2Docker:**
	- Start the Docker Terminal (Boot2Docker)
	- Start a container `$ winpty docker run -ti -p 8080:8080 -v //Users:/uids matbloch/uids //bin/bash`
	- Start the server with `$ python services/v2/online_server.py`
3. Wait till the server is ready for incomming connections
4. Start the user identification client under `bin/app_online_id.exe --port=8080`


### API
Please visit the API readme (`API.md`) for the detailed instructions on how to use the user identification interface.


## Troubleshooting

- The Kinect camera has no connection:
	- Make sure the camera is plugged in and your USB3 controller is supported.
	- You can test your USB controller using the Kinect SDK
- The client application runs very slow:
	- Make sure you compile the client in release mode
- The client can not connect to the server:
	- Make sure you have selected the correct ports when: Starting the Docker container, starting the server, starting the client
	- Make sure you have set the correct port forwarding in your VM when using boot2docker on Windows (see server installation manual)
	- Some applications might block the default port 8080. Check for error message, shutdown other applications or switch ports if needed.