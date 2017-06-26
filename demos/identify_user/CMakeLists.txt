SET(TARGET_SRC
    identify_user.cpp 
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	User
	ImgProc
)

SETUP_APPLICATION(identify_user Demos)
