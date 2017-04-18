#!/usr/bin/env python

"""Global server configuration - Request/Response names match models in src/request src/response"""
ROUTING = {
   "REQUEST": {  # request id range: 0-255
       "ID": {
           "PartialImageIdentificationAligned": 4,  # identify user from image
           "Ping": 222,
           "Disconnect": 223
       },
       "NAME": {
           4: "PartialImageIdentificationAligned",
           222: "Ping",
           223: "Disconnect"
        }
   },
   "RESPONSE": {  # response id range: integer
       "ID": {
           "Identification": 1,
           "PredictionFeedback": 7,
           "OK": 111,
           "Pong": 222,
           "Error": 999
       },
       "NAME": {
           1: "Identification",
           7: "PredictionFeedback",
           111: "OK",
           222: "Pong",
           999: "Error"
       }
   }
}
