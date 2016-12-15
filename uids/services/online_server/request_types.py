#!/usr/bin/python
import response_types as r
# config
from config import ROUTING
r.ROUTING = ROUTING


class CollectEmbeddingsByName:

    def __init__(self, server, conn):

        # receive user nice name
        user_nice_name = server.receive_string(conn)

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # ASSUME USER NAME IS UNIQUE! Picks first user with this name!
        user_id = server.user_db.get_id_from_name(user_nice_name)

        if user_id is None:
            # generate new user id
            print "--- creating new user"
            user_id = server.user_db.create_new_user(user_nice_name)

        # generate embeddings
        embeddings = server.embedding_gen.get_embeddings(images)

        if not embeddings:
            r.Error(server, conn, "No embeddings could be generated off the images")
            return

        # add to database
        server.user_db.add_embeddings(user_id, embeddings)

        r.OK(server, conn)


class CalcEmbedding:

    def __init__(self, server, conn):

        # receive image size
        img_size = server.receive_uint(conn)
        # receive image
        user_face = server.receive_rgb_image(conn, img_size, img_size)
        # generate embedding
        embedding = server.embedding_gen.get_embedding(user_face)

        if embedding is None:
            r.Error(server, conn, "Could not generate face embedding.")
            return

        r.Embedding(server, conn, embedding)


class TrainClassifier:

    def __init__(self, server, conn):
        # train and save classifier
        server.classifier.trigger_training()

        # save user database
        server.user_db.save()

        r.OK(server, conn)


class ImageIdentification:

    def __init__(self, server, conn):

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images)

        if not embeddings:
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # predict user id
        user_id = server.classifier.predict_label_multi(embeddings)

        if user_id is None:
            r.Error(server, conn, "Label could not be predicted - Classifier might not be trained.")
            return

        # TODO: handle unknown detection

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)

        server.user_db.print_users()

        if user_name is None:
            user_name = "unnamed"

        print "--- User ID: " + str(user_id) + " | name: " + user_name

        r.Identification(server, conn, int(user_id), user_name)
