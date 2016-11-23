SET(TARGET_SRC
    skeletalFaceDetection.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
)

SETUP_APPLICATION(skeletal_face_detection Tests)


SET(TARGET_SRC
    faceTracking.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
)

SETUP_APPLICATION(face_tracking Tests)
