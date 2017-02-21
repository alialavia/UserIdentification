SET(TARGET_SRC
    online_identification.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	User
	ImgProc
)

SETUP_APPLICATION(online_id Applications)
