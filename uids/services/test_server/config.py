#!/usr/bin/env python

"""Global server configuration - Request/Response names match models in src/request src/response"""
ROUTING = {
   "REQUEST": {  # request id range: 0-255
       "ID": {
           "Update": 10,
           "Ping": 222

       },
       "NAME": {
           10: "Update",
           222: "Ping"
        }
   },
   "RESPONSE": {  # response id range: integer
       "ID": {
           "UpdateFeedback": 5,
           "OK": 111,
           "Pong": 222,
           "Error": 999
       },
       "NAME": {
           5: "UpdateFeedback",
           111: "OK",
           222: "Pong",
           999: "Error"
       }
   }
}
