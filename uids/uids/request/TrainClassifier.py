#!/usr/bin/python
from uids.response.OK import OK as OKResponse


class TrainClassifier:

    def __init__(self, server, conn):
        # train and save classifier
        server.classifier.trigger_training()

        # save user database
        server.user_db.save()

        OKResponse(server, conn)