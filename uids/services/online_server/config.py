#!/usr/bin/env python

"""Global server configuration - Request/Response names match models in src/request src/response"""
ROUTING = {
   "REQUEST": {
       "ID": {
           "ImageIdentification": 1,  # identify user from image
           "ImageIdentificationUpdate": 2,  # collect embeddings from images for user with ID XY
           "ImageIdentificationAligned": 6,
           "ImageIdentificationUpdateAligned": 7


       },
       "NAME": {
           1: "ImageIdentification",
           2: "ImageIdentificationUpdate",
           6: "ImageIdentificationAligned",
           7: "ImageIdentificationUpdateAligned"
       }
   },
   "RESPONSE": {
       "ID": {
           "Identification": 1,
           "Embedding": 2,
           "OK": 111,
           "Error": 999
       },
       "NAME": {
           1: "Identification",
           2: "Embedding",
           111: "OK",
           999: "Error"
       }
   }
}
