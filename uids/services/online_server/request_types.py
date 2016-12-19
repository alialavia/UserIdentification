#!/usr/bin/python
import response_types as r
# config
from config import ROUTING
r.ROUTING = ROUTING


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
        user_id = server.classifier.predict_class(embeddings)

        if user_id is None:
            r.Error(server, conn, "Label could not be predicted - Face cannot be detected.")
            return
        elif user_id < 0:
            # unknown user
            print "--- creating new user"
            user_id = server.user_db.create_new_user("a_user")
            server.user_db.print_users()
            # add classifier
            server.classifier.init_classifier(user_id, embeddings)

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)

        if user_name is None:
            user_name = "unnamed"

        print "--- User ID: " + str(user_id) + " | name: " + user_name

        r.Identification(server, conn, int(user_id), user_name)


class ImageIdentificationUpdate:

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        print "--------ImageIdentificationUpdate-------------"
        print "    USER ID: " + str(user_id)
        print "----------------------------------------------"

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images)

        if not embeddings:
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # submit data
        server.classifier.process_labeled_stream_data(user_id, embeddings)

        r.OK(server, conn)


class ImageIdentificationAligned:

    def __init__(self, server, conn):

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings:
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # predict user id
        user_id = server.classifier.predict_class(embeddings)

        if user_id is None:
            r.Error(server, conn, "Label could not be predicted - Face cannot be detected.")
            return
        elif user_id < 0:
            # unknown user
            print "--- creating new user"
            user_id = server.user_db.create_new_user("a_user")
            server.user_db.print_users()
            # add classifier
            server.classifier.init_classifier(user_id, embeddings)

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)

        if user_name is None:
            user_name = "unnamed"

        print "--- User ID: " + str(user_id) + " | name: " + user_name

        r.Identification(server, conn, int(user_id), user_name)


class ImageIdentificationUpdateAligned:

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        print "--------ImageIdentificationUpdateAligned------"
        print "    USER ID: " + str(user_id)
        print "----------------------------------------------"

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings:
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # submit data
        server.classifier.process_labeled_stream_data(user_id, embeddings)

        r.OK(server, conn)


class SaveDatabase:

    def __init__(self, server, conn):

        # save user database
        server.user_db.save()

        r.OK(server, conn)
