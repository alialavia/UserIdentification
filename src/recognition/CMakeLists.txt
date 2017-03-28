SET(LIB_NAME Recognition)
SET(HEADER_PATH ${user_identification_SOURCE_DIR}/include/${LIB_NAME})
SET(TARGET_H
    ${HEADER_PATH}/TorsoRecognition.h
    ${HEADER_PATH}/LowerBodyRecognition.h
    ${HEADER_PATH}/BodyParts.h
)

SET(TARGET_SRC
	TorsoRecognition.cpp
	LowerBodyRecognition.cpp
	BodyParts.cpp
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
)

SETUP_LIBRARY(${LIB_NAME})
