#!/usr/bin/env python

"""Global server configuration - Request/Response names match models in src/request src/response"""
ROUTING = {
   "REQUEST": {  # request id range: 0-255
       "ID": {
           "ImageReceival": 1,
           "FeatureGeneration": 2,
           "Ping": 222
       },
       "NAME": {
           1: "ImageReceival",
           2: "FeatureGeneration",
           222: "Ping"
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
           "Pong": 222,
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
           222: "Pong",
           999: "Error"
       }
   }
}
