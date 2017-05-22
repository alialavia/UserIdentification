
# add all folders
FOREACH( mylibfolder
        io
		user
		tracking
		math
		imgproc
		features
		recognition
		gui
		segmentation
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

