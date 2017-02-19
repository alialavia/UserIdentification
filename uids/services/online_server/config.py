#!/usr/bin/env python

"""Global server configuration - Request/Response names match models in src/request src/response"""
ROUTING = {
   "REQUEST": {  # request id range: 0-255
       "ID": {
           "ImageIdentification": 1,  # identify user from image
           "ImageIdentificationPrealigned": 2,
           "Update": 10,  # collect embeddings from images for user with ID XY
           "UpdateRobust": 11,
           "UpdatePrealigned": 12,
           "UpdatePrealignedRobust": 13,
           "ImageAlignment": 22,
           "ProfilePictureUpdate": 23,
           "GetProfilePictures": 24
       },
       "NAME": {
           1: "ImageIdentification",
           2: "ImageIdentificationPrealigned",
           10: "Update",
           11: "UpdateRobust",
           12: "UpdatePrealigned",
           13: "UpdatePrealignedRobust",
           22: "ImageAlignment",
           23: "ProfilePictureUpdate",
           24: "GetProfilePictures"
        }
   },
   "RESPONSE": {  # response id range: integer
       "ID": {
           "Identification": 1,
           "Embedding": 2,
           "Image": 3,
           "QuadraticImage": 4,
           "UpdateFeedback": 5,
           "Reidentification": 10,
           "ProfilePictures": 20,
           "OK": 111,
           "Error": 999
       },
       "NAME": {
           1: "Identification",
           2: "Embedding",
           3: "Image",
           4: "QuadraticImage",
           5: "UpdateFeedback",
           10: "Reidentification",
           20: "ProfilePictures",
           111: "OK",
           999: "Error"
       }
   }
}
