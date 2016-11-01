
# add all folders
FOREACH( mylibfolder
        io
		user
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

