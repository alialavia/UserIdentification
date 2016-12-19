
# add all folders
FOREACH( mylibfolder
        io
		user
		tracking
		math
		imgproc
		features
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

