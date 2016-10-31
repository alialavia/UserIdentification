SET(TARGET_SRC
    tcp_client.cpp 
    # mainsource.cpp	# only this file contains the main() function!
)
SET(TARGET_H
    # header.h
)

SETUP_APPLICATION(tcp_client Tests)
