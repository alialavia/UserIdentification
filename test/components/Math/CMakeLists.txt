SET(TARGET_SRC
    Array3D.cpp 
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Math
	Tracking
)

SETUP_APPLICATION(Array3D Unit Tests)
