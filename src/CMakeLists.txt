
# add all folders
FOREACH( mylibfolder
        io
		# otherlib
    )
	
    ADD_SUBDIRECTORY(${mylibfolder})

ENDFOREACH()

