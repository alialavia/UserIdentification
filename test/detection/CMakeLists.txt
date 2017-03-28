SET(TARGET_SRC
    torsoDetection.cpp
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	Recognition
	Features
)

SETUP_APPLICATION(torso_detection Tests)

# ------------
SET(TARGET_SRC
    lowerBodyDetection.cpp
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	Recognition
	Features
)

SETUP_APPLICATION(lower_body_detection Tests)

# ------------
SET(TARGET_SRC
    skinDetection.cpp
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	Recognition
	Features
)

SETUP_APPLICATION(skin_detection Tests)

# ------------
SET(TARGET_SRC
    bodyPartSegmentation.cpp
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	Recognition
	Features
)

SETUP_APPLICATION(body_part_segmentation Tests)
