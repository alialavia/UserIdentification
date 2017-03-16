SET(TARGET_SRC
    img_converter.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES

)

SETUP_APPLICATION(img_converter Applications)
