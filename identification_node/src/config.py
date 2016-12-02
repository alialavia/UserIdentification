#!/usr/bin/env python

"""Global server configuration - Request/Response names match models in src/request src/response"""
CONFIG = {
    "ROUTING": {
        "REQUEST": {
            "ID": {
                "ImageIdentification": 1,                # identify user from image
                "CollectEmbeddingsByID": 2,      # collect embeddings from images for user with ID XY
                "CollectEmbeddingsByName": 3,    # collect embeddings from images for user with name XY - names assumed to be unique
                "CalcEmbedding": 4,                 # calculate embedding from image
                "TrainClassifier": 5               # trigger classifier training
            },
            "NAME": {
                1: "ImageIdentification",
                2: "CollectEmbeddingsByID",
                3: "CollectEmbeddingsByName",
                4: "CalcEmbedding",
                5: "TrainClassifier"
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
    },
}