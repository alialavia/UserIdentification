SET(TARGET_SRC
    UniqueSetAssoc.cpp 
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Math
	Tracking
)

SETUP_APPLICATION(UniqueSetAssoc Unit Tests)
