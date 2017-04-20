SET(TARGET_SRC
    v2_components.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
)

SETUP_APPLICATION(v2_components V2)

SET(TARGET_SRC
    v2_weights.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	IO
	gflags
	Features
)

SETUP_APPLICATION(v2_weights V2)
