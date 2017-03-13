SET(TARGET_SRC
    feature_generation.cpp 
)
SET(TARGET_H
    # header.h
)

SETUP_APPLICATION(feature_generation Benchmarks)

# --------------------

SET(TARGET_SRC
    face_alignment.cpp 
)
SET(TARGET_H
    # header.h
)
SET(TARGET_EXTERNAL_LIBRARIES
	Tracking
	IO
	Features
)

SETUP_APPLICATION(face_alignment Benchmarks)

