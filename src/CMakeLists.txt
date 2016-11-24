
# add all folders
FOREACH( mylibfolder
        io
		user
		tracking
		math
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

