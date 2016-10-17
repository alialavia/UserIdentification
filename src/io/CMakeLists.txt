SET(LIB_NAME io)
SET(HEADER_PATH ${user_identification_SOURCE_DIR}/include/${LIB_NAME})
SET(TARGET_H
    ${HEADER_PATH}/talker.h	# sample
    ${HEADER_PATH}/kinect.h
)

SET(TARGET_SRC
    talker.cpp	# sample
	kinect.cpp
    # mymain2.cpp
)

# --------------------------------------------
#	Create library
# --------------------------------------------

# link internally (other project libraries)
SET(TARGET_LIBRARIES
    # io
)
# link externally (library paths included over "find_library(${libname})")

# MESSAGE_ESCAPED(${KINECTSDK2_LIBRARIES})

SET(TARGET_EXTERNAL_LIBRARIES
    # ${MATH_LIBRARY}
	${KINECTSDK2_LIBRARIES}	# Kinect SDK 2.0
)

SETUP_LIBRARY(${LIB_NAME})
