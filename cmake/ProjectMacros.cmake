
# link to internal libraries
MACRO(LINK_INTERNAL TRGTNAME)
    TARGET_LINK_LIBRARIES(${TRGTNAME} ${ARGN})
ENDMACRO(LINK_INTERNAL TRGTNAME)

# link to external libraries
MACRO(LINK_EXTERNAL TRGTNAME)
    TARGET_LINK_LIBRARIES(${TRGTNAME} ${ARGN})
ENDMACRO(LINK_EXTERNAL TRGTNAME)

# ---------------------------------------------------------------------
# Setup internal libraries
# ---------------------------------------------------------------------

#######################################################################################################
#	Macro to setup libraries. The following variables need to be defined first:
#	LIB_NAME  					name of the target library
#	TARGET_SRC  				target source
#	TARGET_H 					target headers
#	TARGET_LIBRARIES 		 	internal library dependencies
#	TARGET_EXTERNAL_LIBRARIES 	external library dependencies
#	TARGET_LABEL 				IDE target label
#	TARGET_FOLDER 				IDE solution folder. Default: "Core"
##########################################################################################################

MACRO(SETUP_LIBRARY LIB_NAME)

		# MESSAGE("== SETUP_LIBRARY: Target source: ${TARGET_SRC}")
		# MESSAGE("== SETUP_LIBRARY: Target headers: ${TARGET_H}")

        SET(TARGET_NAME ${LIB_NAME} )
        SET(TARGET_TARGETNAME ${LIB_NAME} )
        ADD_LIBRARY(${LIB_NAME}
            ${DYNAMIC_OR_STATIC_LIBS}		# dynamic/static library
            ${TARGET_H}						# headers
            ${TARGET_SRC}					# sources
        )
		
		# set solution folder
		IF(TARGET_FOLDER)
			SET_TARGET_PROPERTIES(${LIB_NAME} PROPERTIES FOLDER ${TARGET_FOLDER})
		ELSE()
			SET_TARGET_PROPERTIES(${LIB_NAME} PROPERTIES FOLDER "Core")
		ENDIF()
		
		# set IDE target label
        IF(TARGET_LABEL)
            SET_TARGET_PROPERTIES(${TARGET_TARGETNAME} PROPERTIES PROJECT_LABEL "${TARGET_LABEL}")
        ENDIF(TARGET_LABEL)

		# link with other libraries
        IF(TARGET_LIBRARIES)
            LINK_INTERNAL(${LIB_NAME} ${TARGET_LIBRARIES})
        ENDIF()
        IF(TARGET_EXTERNAL_LIBRARIES)
            LINK_EXTERNAL(${LIB_NAME} ${TARGET_EXTERNAL_LIBRARIES})
        ENDIF()

ENDMACRO(SETUP_LIBRARY LIB_NAME)

# ---------------------------------------------------------------------
# setup applications and examples
# ---------------------------------------------------------------------

#######################################################################################################
#	Link libraries to applications/examples
#	TARGET_TARGETNAME  			final executable name (overwrites TARGET_NAME with default prefix)
#	TARGET_LIBRARIES 		 	internal library dependencies
#	TARGET_EXTERNAL_LIBRARIES 	external library dependencies
##########################################################################################################

MACRO(SETUP_LINK_LIBRARIES)

    SET(TARGET_ADDED_LIBRARIES ${TARGET_LIBRARIES})

	# use common libraries
    SET(TARGET_LIBRARIES ${TARGET_COMMON_LIBRARIES})

    FOREACH(LINKLIB ${TARGET_ADDED_LIBRARIES})
		SET(IN_COMMON TRUE)
		
		# check if already in common libs
		FOREACH (value ${TARGET_COMMON_LIBRARIES})
			IF (${value} STREQUAL ${LINKLIB})
			SET(IN_COMMON FALSE)
			ENDIF (${value} STREQUAL ${LINKLIB})
		ENDFOREACH (value ${TARGET_COMMON_LIBRARIES})

		# append if not in common libs
		IF(NOT IN_COMMON)
			LIST(APPEND TARGET_LIBRARIES ${LINKLIB})
		ENDIF(NOT IN_COMMON)
    ENDFOREACH(LINKLIB)

	# link libraries
	LINK_INTERNAL(${TARGET_TARGETNAME} ${TARGET_LIBRARIES})
	LINK_EXTERNAL(${TARGET_TARGETNAME} ${TARGET_EXTERNAL_LIBRARIES})

ENDMACRO(SETUP_LINK_LIBRARIES)


#######################################################################################################
#	Setup command line application
#	TARGET_DEFAULT_PREFIX  					default target prefix (added to all executables)
#	TARGET_DEFAULT_LABEL_PREFIX				default IDE target label
#	TARGET_DEFAULT_FOLDER					default IDE solution folder
#	TARGET_TARGETNAME						final target name (overwrites TARGET_NAME with default prefix)
##########################################################################################################

# general case
MACRO(SETUP_EXE)

	# set target name (executable name)
    IF(NOT TARGET_TARGETNAME)
        SET(TARGET_TARGETNAME "${TARGET_DEFAULT_PREFIX}${TARGET_NAME}")
    ENDIF(NOT TARGET_TARGETNAME)
	
	# set IDE target label
    IF(NOT TARGET_LABEL)
            SET(TARGET_LABEL "${TARGET_DEFAULT_LABEL_PREFIX} ${TARGET_NAME}")
    ENDIF(NOT TARGET_LABEL)

	ADD_EXECUTABLE(${TARGET_TARGETNAME} ${TARGET_SRC} ${TARGET_H})
	
    SET_TARGET_PROPERTIES(${TARGET_TARGETNAME} PROPERTIES PROJECT_LABEL "${TARGET_LABEL}")
    SET_TARGET_PROPERTIES(${TARGET_TARGETNAME} PROPERTIES OUTPUT_NAME ${TARGET_TARGETNAME})
	# add postfix for release versions
    SET_TARGET_PROPERTIES(${TARGET_TARGETNAME} PROPERTIES DEBUG_OUTPUT_NAME "${TARGET_TARGETNAME}${CMAKE_DEBUG_POSTFIX}")
    SET_TARGET_PROPERTIES(${TARGET_TARGETNAME} PROPERTIES RELEASE_OUTPUT_NAME "${TARGET_TARGETNAME}${CMAKE_RELEASE_POSTFIX}")
    SET_TARGET_PROPERTIES(${TARGET_TARGETNAME} PROPERTIES RELWITHDEBINFO_OUTPUT_NAME "${TARGET_TARGETNAME}${CMAKE_RELWITHDEBINFO_POSTFIX}")
    SET_TARGET_PROPERTIES(${TARGET_TARGETNAME} PROPERTIES MINSIZEREL_OUTPUT_NAME "${TARGET_TARGETNAME}${CMAKE_MINSIZEREL_POSTFIX}")

	# link to libraries
    SETUP_LINK_LIBRARIES()

ENDMACRO(SETUP_EXE)

# setup an application
MACRO(SETUP_APPLICATION APPLICATION_NAME FOLDER_NAME)

        SET(TARGET_NAME ${APPLICATION_NAME} )

		# setup command line application
        SETUP_EXE()

		# set solution folder
        SET_TARGET_PROPERTIES(${TARGET_TARGETNAME} PROPERTIES FOLDER ${FOLDER_NAME})
		
		# unset traget name - allow multiple calls to SETUP_EXE
		UNSET(TARGET_TARGETNAME)

ENDMACRO(SETUP_APPLICATION)
