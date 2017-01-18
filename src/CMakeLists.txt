
# add all folders
FOREACH( mylibfolder
        io
		user
		tracking
		math
		imgproc
		features
		recognition
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

