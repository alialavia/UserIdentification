SET(LIB_NAME IO)
SET(HEADER_PATH ${user_identification_SOURCE_DIR}/include/${LIB_NAME})
SET(TARGET_H
    ${HEADER_PATH}/KinectInterface.h
    ${HEADER_PATH}/Networking.h
)

SET(TARGET_SRC
	KinectInterface.cpp
	Networking.cpp
)

# --------------------------------------------
#	Create library
# --------------------------------------------

# link internally (other project libraries)
# SET(TARGET_LIBRARIES

# )
# link externally (library paths included over "find_library(${libname})")
SET(TARGET_EXTERNAL_LIBRARIES
	${KINECTSDK2_LIBRARIES}	# Kinect SDK 2.0
	${OpenCV_LIBS}			# OpenCV 3
	ws2_32					# Windows TCP Sockets
)

SETUP_LIBRARY(${LIB_NAME})
