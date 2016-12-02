#!/usr/bin/python
from src.response.Error import Error as ErrorResponse
from src.response.Identification import Identification as IdentificationResponse


class ImageIdentification:

    def __init__(self, server, conn):

        print "--- Identification"

        # receive image size
        img_size = server.receive_uint(conn)
        # receive image
        user_face = server.receive_rgb_image(conn, img_size, img_size)

        # generate embedding
        embedding = server.embedding_gen.get_embedding(user_face)

        if embedding is None:
            ErrorResponse(server, conn, "Could not generate face embedding.")
            return

        # predict user id
        user_id = server.classifier.predict_label(embedding)

        if user_id is None:
            ErrorResponse(server, conn, "Label could not be predict - Classifier might not be trained.")
            return

        # TODO: handle unknown detection

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)
        if user_name is None:
            user_name = "unnamed"

        print "--- User ID: " + str(user_id) + " | name: " + user_name

        IdentificationResponse(server, conn, int(user_id), user_name)