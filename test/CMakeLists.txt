PROJECT(USER_IDENTIFICATION_TESTS)

SET(USER_IDENTIFICATION_TESTS_DIR ${PROJECT_SOURCE_DIR})

SET(TARGET_DEFAULT_PREFIX "test_")
SET(TARGET_DEFAULT_LABEL_PREFIX "")
# IDE folder
SET(TARGET_DEFAULT_FOLDER "Test applications")

# project libraries for all applications
SET(TARGET_COMMON_LIBRARIES 
    # yourAdditionalLib
	IO
	gflags
)

ADD_SUBDIRECTORY(networking)
ADD_SUBDIRECTORY(tracking)
ADD_SUBDIRECTORY(recognition)
ADD_SUBDIRECTORY(classification)

# component unit testing
ADD_SUBDIRECTORY(components)