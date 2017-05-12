#!/usr/bin/python
import numpy as np
import response_types as r
from uids.utils.Logger import Logger as log
# config
from config import ROUTING
r.ROUTING = ROUTING


# --------------- IDENTIFICATION

class PartialImageIdentificationAligned:

    def __init__(self, server, conn, handle):

        # receive tracking id
        tracking_id = server.receive_uint(conn)

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # get sample poses
        sample_poses = []
        for x in range(0, len(images)):
            pitch = server.receive_char(conn)
            yaw = server.receive_char(conn)
            sample_poses.append([pitch, yaw])
        sample_poses = np.array(sample_poses)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # accumulate samples
        is_save_set, current_samples, _weights_placeholder, current_poses = \
            server.classifier.id_controller.accumulate_samples(
                tracking_id, new_samples=embeddings, sample_poses=sample_poses
            )

        log.info('server', "tracking id: {}".format(tracking_id))

        if len(current_samples) == 0:
            # queue has just been resetted
            r.Error(server, conn, "Samples are inconsistent - starting accumulation again...")
            return

        # predict class similarities
        new_class_guaranteed = server.classifier.is_guaranteed_new_class(current_samples)

        if new_class_guaranteed and len(current_samples) > 2:
            id_pred = -1
            confidence = 100
            is_consistent = True    # ??
            is_save_set = True
        else:
            # do meta recognition and predict the user id from the cls scores
            is_consistent, id_pred, confidence = server.classifier.predict_class(current_samples, sample_poses=current_poses)

            # convert to integer
            confidence = int(confidence*100.0)

        # get user nice name
        user_name = server.user_db.get_name_from_id(id_pred)

        print ".... is_save: {}, is_consistent: {}, id_pred: {}, confidence: {}".format(is_save_set, is_consistent, id_pred, confidence)

        if user_name is None:
            user_name = "unnamed"

        if is_save_set:
            # SAVE SET - TAKE ACTION
            profile_picture = None

            if is_consistent:
                # new identity
                if id_pred < 0:
                    # unknown user
                    print "--- creating new user"
                    log.info('db', "Creating new User")
                    user_id = server.user_db.create_new_user("a_user")
                    server.user_db.print_users()
                    # add classifier
                    server.classifier.init_new_class(user_id, current_samples, sample_poses=current_poses)
                    id_pred = user_id
                else:
                    # for s in current_samples:
                    #     print "s: {:.2f}".format(s[0])
                    # add data for training and return identification
                    # add to data model
                    server.classifier.data_controller.add_samples(user_id=id_pred, new_samples=current_samples, new_poses=current_poses)
                    # add to classifier training queue
                    server.classifier.add_training_data(id_pred, current_samples)

                    # get profile picture
                    profile_picture = server.user_db.get_profile_picture(id_pred)

                # cleanup
                server.classifier.id_controller.drop_samples(tracking_id)

                # valid identification
                log.info('server', "User identification complete: {} [ID], {} [Username]".format(id_pred, user_name))
                r.Identification(server, conn, int(id_pred), user_name, confidence=confidence,
                                 profile_picture=profile_picture)
            else:
                # inconsistent prediction - samples might be bad. dump and take new samples

                server.classifier.id_controller.drop_samples(tracking_id)
                r.Error(server, conn, "Samples are inconsistent - starting accumulation again...")
                return
                # TODO: is feedback useful here?

        else:
            # UNSAVE SET - WAIT TILL SAVE SET IS ACCUMULATED

            # return prediction and confidence - but no identification
            r.PredictionFeedback(server, conn, int(id_pred), user_name, confidence=confidence)


# --------------- UPDATE

class PartialUpdateAligned:

    def __init__(self, server, conn, handle):
        # receive user id
        user_id = server.receive_uint(conn)

        log.info('server', 'User Update (Aligned, Robust) for ID {}'.format(user_id))

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        # get sample poses
        sample_poses = []
        for x in range(0, len(images)):
            pitch = server.receive_char(conn)
            yaw = server.receive_char(conn)
            sample_poses.append([pitch, yaw])
        sample_poses = np.array(sample_poses)

        print sample_poses

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, align=False)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # TODO: calculate weights
        weights = np.repeat(10, len(images))

        # accumulate samples - check for inconsistencies
        verified_data, verified_poses, reset_user, id_pred, confidence = server.classifier.update_controller.accumulate_samples(
            user_id, embeddings, sample_weights=weights, sample_poses=sample_poses
        )

        log.info('cl', "verified_data (len: {}), reset_user: {}: ID {}, conf {}".format(len(verified_data), reset_user, id_pred, confidence))

        # forward save part of data
        if verified_data.size:
            # for s in embeddings:
            #     print "new: {:.8f}".format(s[0])
            # print "------------------"
            # for s in verified_data:
            #     print "s: {:.5f}".format(s[0])

            # add to data model
            server.classifier.data_controller.add_samples(user_id=user_id, new_samples=verified_data, new_poses=verified_poses)
            # add to classifier training queue
            server.classifier.add_training_data(user_id, verified_data)

        # reset user if queue has become inconsistent or wrong user is predicted
        if reset_user:
            log.severe("USER VERIFICATION FAILED - FORCE REIDENTIFICATION")
            r.Reidentification(server, conn)
            return

        # return prediction feedback
        user_name = server.user_db.get_name_from_id(id_pred)
        if user_name is None:
            user_name = "unnamed"
        r.PredictionFeedback(server, conn, id_pred, user_name, confidence=int(confidence*100.0))


class ImageIdentificationPrealignedCS:

    def __init__(self, server, conn, handle):

        nr_users = server.receive_uint(conn)
        target_users = []
        for x in range(0, nr_users):
            # get target class ids (uint)
            user_id = server.receive_uint(conn)
            target_users.append(user_id)

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        log.severe('ImageIdentificationPrealignedCS')

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        if -1 in target_users:
            # open set user id prediction
            current_weights = np.repeat(1, len(embeddings))
            is_consistent, user_id, confidence = server.classifier.predict_class(embeddings, current_weights)
        else:
            # closed set user id prediction
            user_id = server.classifier.predict_closed_set(target_users, embeddings)

            if user_id is None:
                r.Error(server, conn, "Label could not be predicted - Samples are contradictory.")
                return

        # get user nice name
        user_name = server.user_db.get_name_from_id(user_id)

        if user_name is None:
            user_name = "unnamed"

        # get profile picture
        profile_picture = server.user_db.get_profile_picture(user_id)
        log.info('server', "User identification complete: {} [ID], {} [Username]".format(user_id, user_name))
        r.Identification(server, conn, int(user_id), user_name,
                         profile_picture=profile_picture)

# --------------- MISC


class CancelIdentification:

    def __init__(self, server, conn, handle):
        # receive tracking id
        tracking_id = server.receive_uint(conn)

        log.info('server', "Dropping identification queue for tracking id {}".format(tracking_id))
        # drop samples
        server.classifier.id_controller.drop_samples(tracking_id=tracking_id)
        r.OK(server, conn)


class SaveDatabase:

    def __init__(self, server, conn, handle):

        # save user database
        server.user_db.save()

        r.OK(server, conn)


class ImageAlignment:

    def __init__(self, server, conn, handle):

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


class ProfilePictureUpdate:

    def __init__(self, server, conn, handle):
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
        # user_id_predicted = server.classifier.predict(embedding)

        # # check if correct user
        # if user_id_predicted is None:
        #     r.Error(server, conn, "Label could not be predicted - Face is unambiguous.")
        #     return
        # elif user_id_predicted != user_id:
        #     # unknown user
        #     r.Error(server, conn, "The profile image does not come from the same person!")
        #     return

        server.user_db.set_profile_picture(user_id, image)

        # send back image
        r.QuadraticImage(server, conn, image)


class GetProfilePictures:

    def __init__(self, server, conn, handle):

        log.info('server', 'Getting all profile pictures')

        uids, pictures = server.user_db.get_all_profile_pictures()

        # send back image
        r.ProfilePictures(server, conn, uids, pictures)


class Ping:

    def __init__(self, server, conn, handle):
        r.Pong(server, conn)


class Disconnect:
    def __init__(self, server, conn, handle):
        # break connection
        handle[0] = False
