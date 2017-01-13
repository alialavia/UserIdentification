PROJECT(USER_IDENTIFICATION_APPLICATIONS)

SET(USER_IDENTIFICATION_APPLICATION_DIR ${PROJECT_SOURCE_DIR})

SET(TARGET_DEFAULT_PREFIX "app_")
SET(TARGET_DEFAULT_LABEL_PREFIX "")
# IDE folder
SET(TARGET_DEFAULT_FOLDER "Applications")

# project libraries for all applications
SET(TARGET_COMMON_LIBRARIES 
    # yourAdditionalLib
	Tracking
	IO
	User
	gflags
	ImgProc
)

ADD_SUBDIRECTORY(face_captures)
ADD_SUBDIRECTORY(auto_classifier_training)
ADD_SUBDIRECTORY(embeddings_from_images)
ADD_SUBDIRECTORY(offline_identification)
ADD_SUBDIRECTORY(online_identification)
ADD_SUBDIRECTORY(image_labeler)
