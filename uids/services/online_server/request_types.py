#!/usr/bin/python
import response_types as r
from uids.utils.Logger import Logger as log
# config
from config import ROUTING
r.ROUTING = ROUTING


class ProfilePictureUpdate:

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        log.info('server', 'Updating profile picture for user with ID {}'.format(user_id))

        # receive images
        image = server.receive_image_squared(conn)

        # generate embedding
        embedding = server.embedding_gen.get_embeddings([image])

        if not embedding.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # predict user id
        user_id_predicted = server.classifier.predict(embedding)

        # check if correct user
        if user_id_predicted is None:
            r.Error(server, conn, "Label could not be predicted - Face cannot be detected.")
            return
        elif user_id_predicted != user_id:
            # unknown user
            r.Error(server, conn, "The profile image does not come from the same person!")
            return

        server.user_db.set_profile_picture(user_id, image)

        # send back image
        r.QuadraticImage(server, conn, image)


class ImageIdentification:

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
        elif user_id < 0:
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

        r.Identification(server, conn, int(user_id), user_name, profile_picture=profile_picture)


class ImageIdentificationUpdate:

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        log.info('server', 'User Update (Robust) for ID {}'.format(user_id))

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images)

        if embeddings.shape[0] < 5:
            r.Error(server, conn, "Not enough images for update quality check.")
            return

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        log.info('cl', "Starting to process stream data...")

        # submit data
        succ = server.classifier.process_labeled_stream_data(user_id, embeddings)

        if succ:
            r.OK(server, conn)
        else:
            # update was not classified as the labeled class
            # force reidentification
            r.Reidentification(server, conn)


class ImageIdentificationAligned:

    def __init__(self, server, conn):

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # predict user id
        user_id = server.classifier.predict(embeddings)

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
    """
    Pure Update
    - Assumes given label/id is correct
    """

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        log.info('server', 'User Update (Aligned) for ID {}'.format(user_id))

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        log.info('cl', "Start to process stream data...")

        # submit data (ignore if prediction of samples does not fit model)
        succ = server.classifier.process_labeled_stream_data(user_id, embeddings)

        r.OK(server, conn)


class ImageIdentificationUpdateAlignedRobust:
    """
    Pure Update
    - Checks
    """

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        log.info('server', 'User Update (Aligned, Robust) for ID {}'.format(user_id))

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        log.info('cl', "Starting to process stream data...")

        # submit data
        succ = server.classifier.process_labeled_stream_data(user_id, embeddings)

        if succ:
            r.OK(server, conn)
        else:
            # update was not classified as the labeled class
            # force reidentification
            r.Reidentification(server, conn)


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
