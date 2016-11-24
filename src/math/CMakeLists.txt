SET(LIB_NAME Math)
SET(HEADER_PATH ${user_identification_SOURCE_DIR}/include/${LIB_NAME})
SET(TARGET_H
    ${HEADER_PATH}/Math.h
)

SET(TARGET_SRC
	Math.cpp
)

# --------------------------------------------
#	Create library
# --------------------------------------------

# link externally (library paths included over "find_library(${libname})")
SET(TARGET_EXTERNAL_LIBRARIES
	${OpenCV_LIBS}			# OpenCV 3
)

SETUP_LIBRARY(${LIB_NAME})
