#!/usr/bin/env python

"""Global server configuration - Request/Response names match models in src/request src/response"""
CONFIG = {
    "ROUTING": {
        "REQUEST": {
            "ID": {
                "Identification": 1,                # identify user from image
                "CollectEmbeddingsByID": 2,      # collect embeddings from images for user with ID XY
                "CollectEmbeddingsByName": 3,    # collect embeddings from images for user with name XY - names assumed to be unique
                "GetEmbedding": 4,                 # calculate embedding from image
                "TrainClassifier": 5               # trigger classifier training
            },
            "NAME": {
                1: "Identification",
                2: "CollectEmbeddingsByID",
                3: "CollectEmbeddingsByName",
                4: "GetEmbedding",
                5: "TrainClassifier"
            }
        },
        "RESPONSE": {
            "ID": {
                "Identification": 1,
                "GetEmbedding": 2,
                "OK": 111,
                "Error": 999
            },
            "NAME": {
                1: "Identification",
                2: "GetEmbedding",
                111: "OK",
                999: "Error"
            }
        }
    },
}