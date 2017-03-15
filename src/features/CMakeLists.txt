SET(LIB_NAME Features)
SET(HEADER_PATH ${user_identification_SOURCE_DIR}/include/${LIB_NAME})
SET(TARGET_H
    ${HEADER_PATH}/Face.h
    ${HEADER_PATH}/Skin.h
)

SET(TARGET_SRC
	Face.cpp
	Skin.cpp
)

# --------------------------------------------
#	Create library
# --------------------------------------------

# link internally (other project libraries)
SET(TARGET_LIBRARIES
	Tracking
)
# link externally (library paths included over "find_library(${libname})")
SET(TARGET_EXTERNAL_LIBRARIES
		${OpenCV_LIBS}			# OpenCV 3
		dlib
)

SETUP_LIBRARY(${LIB_NAME})
