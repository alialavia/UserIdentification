
# add all folders
FOREACH( mylibfolder
        io
		user
		tracking
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

