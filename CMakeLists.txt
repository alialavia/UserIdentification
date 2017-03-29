# User identification library
#
# Project constants:
# PROJECT_BINARY_DIR: Full path to build directory for project
# PROJECT_SOURCE_DIR: Top level source directory for the current project
#
# Userdefined:
# DYNAMIC_OR_STATIC_LIBS
# ----------------------------------------------

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project (user_identification)

# USER_IDENTIFICATION_VERSION: <major>.<minor>.<patch>
SET(USER_IDENTIFICATION_MAJOR_VERSION 0) # major version
SET(USER_IDENTIFICATION_MINOR_VERSION 1)	# feature version
SET(USER_IDENTIFICATION_PATCH_VERSION 0) # patch version
SET(USER_IDENTIFICATION_VERSION ${USER_IDENTIFICATION_MAJOR_VERSION}.${USER_IDENTIFICATION_MINOR_VERSION}.${USER_IDENTIFICATION_PATCH_VERSION})

# Change the default build type to Release
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None, Debug, Release." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Set cmake module search path
set(CMAKE_MODULE_PATH 
	${CMAKE_MODULE_PATH} 
	${CMAKE_SOURCE_DIR}/cmake
)


# build dynamic/static libraries
# [STATIC | SHARED | MODULE]
SET(DYNAMIC_OR_STATIC_LIBS STATIC)

# enable IDE solution folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON) 

#-------------------------------------------------
# Add Submodules
#-------------------------------------------------

add_subdirectory(${user_identification_SOURCE_DIR}/externals)

#-------------------------------------------------
# Include Headers
#-------------------------------------------------

INCLUDE_DIRECTORIES(
    ${user_identification_SOURCE_DIR}/include	# project headers
	${user_identification_SOURCE_DIR}/externals/dlib	# dlib
)

#-------------------------------------------------
# Options
#-------------------------------------------------

# Applications
OPTION(BUILD_USER_IDENTIFICATION_APPLICATIONS "Enable to build Applications" ON)

# Examples
OPTION(BUILD_USER_IDENTIFICATION_EXAMPLES "Enable to build Examples" OFF)

# Tests
OPTION(BUILD_USER_IDENTIFICATION_TESTS "Enable to build Tests" ON)

# Benchmarks
OPTION(BUILD_USER_IDENTIFICATION_BENCHMARKS "Enable to build Benchmarks" ON)

# Folder structure
set(PROJECT_RESOURCES_PATH "resources" CACHE STRING "Resources path")
set(PROJECT_MODELS_PATH "models" CACHE STRING "Models path")

#-------------------------------------------------
# 3rd Party Dependency
#-------------------------------------------------

# Kinect for Windows SDK 2
FIND_PACKAGE(KinectSDK2 REQUIRED)
# include headers
INCLUDE_DIRECTORIES(${KINECTSDK2_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV 3.0 REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11

# include this source when dlib is used
find_package(dlib REQUIRED)

# Alternative:
# Include only headers and add source.cpp to source of target
# SET(DLIB_SOURCE ${user_identification_SOURCE_DIR}/externals/dlib/dlib/all/source.cpp)

# google gflags
find_package(gflags 2.2 REQUIRED)
# include_directories(${GFLAGS_INCLUDE_DIR})

# Platform specific:
IF(WIN32)
ENDIF()

IF(APPLE)

ENDIF()

# Include macro utilities
INCLUDE(ProjectMacros)
#-------------------------------------------------
# Optional Build Components
#-------------------------------------------------

# optional example related dependencies
IF   (BUILD_USER_IDENTIFICATION_EXAMPLES)
	# use find_package() here
ENDIF()

#-------------------------------------------------
# Setup Output Folders
#-------------------------------------------------

SET(OUTPUT_BINDIR ${PROJECT_BINARY_DIR}/bin)		# builddir/bin
SET(OUTPUT_LIBDIR ${PROJECT_BINARY_DIR}/lib)		# builddir/lib

# create bin and lib folders
MAKE_DIRECTORY(${OUTPUT_BINDIR})		# builddir/bin
MAKE_DIRECTORY(${OUTPUT_LIBDIR})		# builddir/lib

# Generic no-config case (e.g. with mingw)

# 1. CMAKE_RUNTIME_OUTPUT_DIRECTORY
# Executables are always treated as runtime targets
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BINDIR})		# builddir/bin

# 2. CMAKE_ARCHIVE_OUTPUT_DIRECTORY
# Static libraries are always treated as archive targets
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_LIBDIR})

# 3. CMAKE_LIBRARY_OUTPUT_DIRECTORY
# Module libraries are always treated as library targets

IF(WIN32)
	# (.dll) output must bin in /bin folder (together with .exe) in order for Windows to find it
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_BINDIR})
ELSE(WIN32)
	# (.a) otherwise place them in lib folder
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_LIBDIR})
ENDIF(WIN32)

# For non-DLL platforms shared libraries are treated as library targets. 
# For DLL platforms the DLL part of a shared library is treated as a runtime target and 
# the corresponding import library is treated as an archive target (.lib). 
# All Windows-based systems including Cygwin are DLL platforms.

FOREACH(CONF ${CMAKE_CONFIGURATION_TYPES})        # For each configuration (Debug, Release, Minsizerel, ... )
	STRING(TOUPPER "${CONF}" CONF)
	SET("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF}" "${OUTPUT_BINDIR}")
	SET("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF}" "${OUTPUT_LIBDIR}")
	IF(WIN32)
		# Windows looks for the .dll in the same folder as the .exe, which goes in the /bin folder
		SET("CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF}" "${OUTPUT_BINDIR}")
	ELSE()
		SET("CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF}" "${OUTPUT_LIBDIR}")
	ENDIF()
ENDFOREACH()

#-------------------------------------------------
# User Options
#-------------------------------------------------

# Expose CMAKE_INCLUDE_PATH and CMAKE_LIBARY_PATH to the GUI so users
# may set these values without needing to manipulate the environment.
SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} CACHE STRING "You may add additional search paths here. Use ; to separate multiple paths.")
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} CACHE STRING "You may add additional search paths here. Use ; to separate multiple paths.")
			   
#-------------------------------------------------
# Preprocessor Directives
#-------------------------------------------------

# debugging preprocessor
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG") 

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# all errors
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
#SET(CMAKE_VERBOSE_MAKEFILE ON)

# use facegrid recording (pose dependent image capturing)
add_definitions(-DFACEGRID_RECORDING)
# check tracking confidence (reidentification in critical situations)
# allows custom "uncertain tracking" state, which blocks new updates upon "clean" robust update
add_definitions(-D_CHECK_TRACKING_CONF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")

else()
	# dlib face detection only runs fast in relase mode
	# use dlib to prealign the face images before they are beeing sent to the server
	add_definitions(-D_DLIB_PREALIGN)
endif()

# always do robust updates (chances are that reinitialization is triggered too often)
add_definitions(-D_ALWAYS_DO_SAFE_UPDATE)

add_definitions(-D_CLOSED_SET_REVALIDATION)


#-------------------------------------------------
# Config Header
#-------------------------------------------------

# generate config file
SET(CONFIG_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h")
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/src/Config.cmake.in"
               "${CONFIG_HEADER}")

#-------------------------------------------------
# Make Path Structure
#-------------------------------------------------   

# create directories
MAKE_DIRECTORY("${OUTPUT_BINDIR}/${PROJECT_RESOURCES_PATH}")
MAKE_DIRECTORY("${OUTPUT_BINDIR}/${PROJECT_MODELS_PATH}")
MAKE_DIRECTORY("${OUTPUT_BINDIR}/${PROJECT_OUTPUT_PATH}")

# copy models
file(COPY ${PROJECT_MODELS_PATH} DESTINATION ${OUTPUT_BINDIR})
Message(${OUTPUT_BINDIR})

#-------------------------------------------------
# Add Sources
#-------------------------------------------------

# Core
ADD_SUBDIRECTORY(src)

# Applications
IF   (BUILD_USER_IDENTIFICATION_APPLICATIONS)
    ADD_SUBDIRECTORY(applications)
ENDIF()

# Examples
IF   (BUILD_USER_IDENTIFICATION_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()

# Tests
IF   (BUILD_USER_IDENTIFICATION_TESTS)
    ADD_SUBDIRECTORY(test)
ENDIF()

# Tests
IF   (BUILD_USER_IDENTIFICATION_BENCHMARKS)
    ADD_SUBDIRECTORY(benchmarking)
ENDIF()

