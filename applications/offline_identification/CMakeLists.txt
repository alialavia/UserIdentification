SET(TARGET_SRC
    offline_identification.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	User
)

SETUP_APPLICATION(offline_id Applications)
