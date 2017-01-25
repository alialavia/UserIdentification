SET(TARGET_SRC
    online_emu.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SET(TARGET_EXTERNAL_LIBRARIES

)

SETUP_APPLICATION(online_emu Applications)
