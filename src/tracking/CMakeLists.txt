SET(LIB_NAME Tracking)
SET(HEADER_PATH ${user_identification_SOURCE_DIR}/include/${LIB_NAME})
SET(TARGET_H
    ${HEADER_PATH}/SkeletonTracker.h
    ${HEADER_PATH}/FaceTracker.h
)

SET(TARGET_SRC
	SkeletonTracker.cpp
	FaceTracker.cpp
)

# --------------------------------------------
#	Create library
# --------------------------------------------

# link internally (other project libraries)
# SET(TARGET_LIBRARIES

# )
# link externally (library paths included over "find_library(${libname})")
SET(TARGET_EXTERNAL_LIBRARIES
		${OpenCV_LIBS}			# OpenCV 3
		Math
		dlib
)

SETUP_LIBRARY(${LIB_NAME})
