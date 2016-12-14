#!/usr/bin/python
from uids.response.Error import Error as ErrorResponse
from uids.response.Identification import Identification as IdentificationResponse


class ImageIdentification:

    def __init__(self, server, conn):

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images)

        if not embeddings:
            ErrorResponse(server, conn, "Could not generate face embeddings.")
            return

        # predict user id
        user_id = server.classifier.predict_label_multi(embeddings)

        if user_id is None:
            ErrorResponse(server, conn, "Label could not be predict - Classifier might not be trained.")
            return

        # TODO: handle unknown detection

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)

        server.user_db.print_users()

        if user_name is None:
            user_name = "unnamed"

        print "--- User ID: " + str(user_id) + " | name: " + user_name

        IdentificationResponse(server, conn, int(user_id), user_name)