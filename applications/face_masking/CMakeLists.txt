SET(TARGET_SRC
    face_masking.cpp 
)
SET(TARGET_H
    # header.h
)


SET(TARGET_EXTERNAL_LIBRARIES
	Segmentation
)

SETUP_APPLICATION(face_masking Applications)
