
# add all folders
FOREACH( mylibfolder
        io
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

