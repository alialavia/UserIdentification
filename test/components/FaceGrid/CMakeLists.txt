SET(TARGET_SRC
    FaceGrid.cpp 
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Math
	Tracking
	User
)

SETUP_APPLICATION(FaceGrid Unit Tests)
