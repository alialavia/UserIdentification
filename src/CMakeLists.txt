
# add all folders
FOREACH( mylibfolder
        io
		user
		tracking
		math
		imgproc
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

