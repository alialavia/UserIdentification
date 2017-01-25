#!/usr/bin/python
import response_types as r
from uids.utils.Logger import Logger as log
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
            log.info('db', "Creating new user")
            user_id = server.user_db.create_new_user("a_user")
            server.user_db.print_users()
            # add classifier
            server.classifier.init_classifier(user_id, embeddings)

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)

        if user_name is None:
            user_name = "unnamed"

        log.info('server', "User identification complete: {} [ID], {} [Username]".format(user_id, user_name))

        r.Identification(server, conn, int(user_id), user_name)


class ImageIdentificationUpdate:

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        log.info('server', 'User Update for ID {}'.format(user_id))

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
            log.info('db', "Creating new User")
            user_id = server.user_db.create_new_user("a_user")
            server.user_db.print_users()
            # add classifier
            server.classifier.init_classifier(user_id, embeddings)

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)

        if user_name is None:
            user_name = "unnamed"

        log.info('server', "User identification complete: {} [ID], {} [Username]".format(user_id, user_name))

        r.Identification(server, conn, int(user_id), user_name)


class ImageIdentificationUpdateAligned:

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        log.info('server', 'User Update for ID {}'.format(user_id))

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings:
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        log.info('cl', "Start to process stream data...")

        # submit data
        server.classifier.process_labeled_stream_data(user_id, embeddings)

        r.OK(server, conn)


class SaveDatabase:

    def __init__(self, server, conn):

        # save user database
        server.user_db.save()

        r.OK(server, conn)


class ImageAlignment:

    def __init__(self, server, conn):

        log.info('server', "Image alignment")

        # receive image
        img = server.receive_rgb_image_squared(conn)

        # align image
        # innerEyesAndBottomLip, outerEyesAndNose
        aligned = server.embedding_gen.align_face(img, 'outerEyesAndNose', 96)

        if aligned is None:
            r.Error(server, conn, "Could not align the image")
            return

        # send aligned image back
        r.QuadraticImage(server, conn, aligned)
