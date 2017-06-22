SET(TARGET_SRC
    client.cpp 
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	User
	ImgProc
)

SETUP_APPLICATION(client Demos)
