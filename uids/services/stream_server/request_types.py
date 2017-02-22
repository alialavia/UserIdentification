#!/usr/bin/python
import response_types as r
from uids.utils.Logger import Logger as log
# config
from config import ROUTING
r.ROUTING = ROUTING


# --------------- IDENTIFICATION

class ImageStreamIdentification:

    def __init__(self, server, conn):

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return



        # predict user id
        user_id = server.classifier.predict(embeddings)

        if user_id is None:
            r.Error(server, conn, "Label could not be predicted - Face cannot be detected.")
            return

        # calculate confidence
        confidence = int(server.classifier.prediction_proba(user_id)*100)

        if user_id < 0:
            # unknown user
            log.info('db', "Creating new user")
            user_id = server.user_db.create_new_user("a_user")
            server.user_db.print_users()
            # add new classifier
            server.classifier.init_classifier(user_id, embeddings)

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)

        if user_name is None:
            user_name = "unnamed"

        # get profile picture
        profile_picture = server.user_db.get_profile_picture(user_id)
        log.info('server', "User identification complete: {} [ID], {} [Username]".format(user_id, user_name))
        r.Identification(server, conn, int(user_id), user_name, confidence=confidence, profile_picture=profile_picture)
