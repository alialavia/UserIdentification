SET(TARGET_SRC
    FaceProfiling.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
)

SETUP_APPLICATION(face_profiling Tests)
