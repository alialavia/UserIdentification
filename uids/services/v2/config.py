#!/usr/bin/env python

"""Global server configuration - Request/Response names match models in src/request src/response"""
ROUTING = {
   "REQUEST": {  # request id range: 0-255
       "ID": {
           "PartialImageIdentificationAligned": 4,  # identify user from image
           "PartialUpdateAligned": 15,
           "ProfilePictureUpdate": 23,
           "CancelIdentification": 221,
           "Ping": 222,
           "Disconnect": 223
       },
       "NAME": {
           4: "PartialImageIdentificationAligned",
           15: "PartialUpdateAligned",
           23: "ProfilePictureUpdate",
           221: "CancelIdentification",
           222: "Ping",
           223: "Disconnect"
        }
   },
   "RESPONSE": {  # response id range: integer
       "ID": {
           "Identification": 1,
            "QuadraticImage": 4,
           "PredictionFeedback": 7,
           "Reidentification": 10,
           "OK": 111,
           "Pong": 222,
           "Error": 999
       },
       "NAME": {
           1: "Identification",
           4: "QuadraticImage",
           7: "PredictionFeedback",
           10: "Reidentification",
           111: "OK",
           222: "Pong",
           999: "Error"
       }
   }
}
