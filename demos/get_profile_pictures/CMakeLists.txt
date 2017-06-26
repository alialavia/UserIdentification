SET(TARGET_SRC
    get_profile_pictures.cpp 
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	User
	ImgProc
)

SETUP_APPLICATION(get_profile_pictures Demos)
