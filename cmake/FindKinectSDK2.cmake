# CMake Module to find and include Kinect drivers
# If the libraries have been found, following variables will be set:
# =======================================================================
# KINECTSDK2_FOUND
# KINECTSDK2_DIR			path to library
# KINECTSDK2_INCLUDE_DIRS	required includes
# KINECTSDK2_LIBRARY_DIRS	library root path
# KINECTSDK2_LIBRARIES		paths to libraries - use with target_link_libraries(target ${KINECTSDK2_LIBRARIES})
# KINECTSDK2_DEFINITIONS
# =======================================================================
# TODO: - Options to select library components
# 		- Support OpenKinect drivers

set(KINECTSDK2_FOUND FALSE BOOL "Kinect 2.x SDK found")
set(KINECTSDK2_DIR "NOT FOUND" PATH "Kinect 2.x SDK path")
set(KINECTSDK2_INCLUDE_DIRS "NOT FOUND" PATH "Kinect 2.x SDK include path")
set(KINECTSDK2_LIBRARY_DIRS "NOT FOUND" PATH "Kinect 2.x SDK library path")
set(KINECTSDK2_LIBRARIES "NOT FOUND" PATH "Kinect 2.x SDK libraries")

if(WIN32)
	# kinect sdk is installed (path saved environment variable)
    if(EXISTS $ENV{KINECTSDK20_DIR})
		
		# kinect sdk found
        set(KINECTSDK2_FOUND TRUE)
        set(KINECTSDK2_DIR $ENV{KINECTSDK20_DIR})
		# escape backward slashes
		string(REPLACE "\\" "/" KINECTSDK2_DIR ${KINECTSDK2_DIR})

		# check for valid visual studio version
		if (MSVC)
			if(MSVC_VERSION LESS 1800)	# 1800 = VS 12.0
			  message(WARNING "== Kinect for Windows SDK v2 is only supported in MSVC >= 12.")
			  set(KINECTSDK2_FOUND FALSE)
			endif()
		endif(MSVC)
		
		# includes
		if(KINECTSDK2_FOUND)
			set(KINECTSDK2_INCLUDE_DIRS ${KINECTSDK2_DIR}/inc)
			if(NOT EXISTS ${KINECTSDK2_INCLUDE_DIRS})
				message("Kinect inc dir not found under: ${KINECTSDK2_INCLUDE_DIRS}")
				set(KINECTSDK2_FOUND FALSE)
			endif()
		endif()

		# library
		if(KINECTSDK2_FOUND)
			# get platform
			set(TARGET_PLATFORM "Platform not detected")
			if(NOT CMAKE_CL_64)
			  set(TARGET_PLATFORM x86)
			else()
			  set(TARGET_PLATFORM x64)
			endif()
		
			set(KINECTSDK2_LIBRARY_DIRS ${KINECTSDK2_DIR}Lib/${TARGET_PLATFORM})
			
			if(NOT EXISTS ${KINECTSDK2_LIBRARY_DIRS})
				message("== Kinect library dir not found under: ${KINECTSDK2_LIBRARY_DIRS}")
				set(KINECTSDK2_FOUND FALSE)
			else()	
				set(KINECTSDK2_LIBRARIES 
					${KINECTSDK2_LIBRARY_DIRS}/Kinect20.lib;
					# ${KinectSDK2_LIBRARY_DIRS}/Kinect20.Face.lib;
					# ${KinectSDK2_LIBRARY_DIRS}/Kinect20.Fusion.lib;
					# ${KinectSDK2_LIBRARY_DIRS}/Kinect20.VisualGestureBuilder.lib
				)
				
				# check if libraries exist
				foreach(target_lib ${KINECTSDK2_LIBRARIES})
					if(NOT EXISTS ${target_lib})
						message("== Library does not exist: ${target_lib}")
						set(KINECTSDK2_FOUND FALSE)
					endif()
				endforeach()
			endif()
		endif(KINECTSDK2_FOUND)
    endif()
else(WIN32)
	# drivers for other platforms (e.g. OpenKinect/freenect)
	# TODO: add cross-platform support
endif()

if(KINECTSDK2_FOUND)
	message("== Kinect SDK dir found in: ${KINECTSDK2_DIR}")
else()
	message("== Kinect SDK dir not found in: ${KINECTSDK2_DIR}")
endif()
