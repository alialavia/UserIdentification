SET(TARGET_SRC
    User.cpp 
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Math
	Tracking
	User
)

SETUP_APPLICATION(User Unit Tests)
