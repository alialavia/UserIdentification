SET(TARGET_SRC
    offline_training.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SETUP_APPLICATION(offline_training Applications)
