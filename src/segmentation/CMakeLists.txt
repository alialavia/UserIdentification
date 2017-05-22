SET(LIB_NAME Segmentation)
SET(HEADER_PATH ${user_identification_SOURCE_DIR}/include/${LIB_NAME})
SET(TARGET_H
    ${HEADER_PATH}/connectedcomponents.h
    ${HEADER_PATH}/facemasker.h
    ${HEADER_PATH}/headsegmenter.h
)

SET(TARGET_SRC
	connectedcomponents.cpp
	facemasker.cpp
	headsegmenter.cpp
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
)

SETUP_LIBRARY(${LIB_NAME})
