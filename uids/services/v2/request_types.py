#!/usr/bin/python
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

        # get weights
        weights = server.receive_uchar_array(conn)

        # generate embedding
        embeddings = server.embedding_gen.get_embeddings(images, False)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        # accumulate samples (optional)
        is_save_set, current_samples, current_weights = server.classifier.id_controller.accumulate_samples(tracking_id, new_samples=embeddings, sample_weights=weights)

        print "tracking id: {}, sample weights: {}".format(tracking_id, weights)

        if len(current_samples) == 0:
            # queue has just been resetted
            r.Error(server, conn, "Samples are inconsistent - starting accumulation again...")
            return

        # predict class similarities
        new_class_guaranteed = server.classifier.is_guaranteed_new_class(current_samples)

        if new_class_guaranteed:
            id_pred = -1
            confidence = 100
            is_consistent = True
        else:
            # do meta recognition and predict the user id from the cls scores
            is_consistent, id_pred, confidence = server.classifier.predict_class(current_samples, current_weights)

            # convert to integer
            confidence = int(confidence*100.0)

        # get user nice name
        user_name = server.user_db.get_name_from_id(id_pred)

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
                    server.classifier.init_new_class(user_id, embeddings)
                else:
                    # add data for training and return identification
                    server.update_controller.add_samples_for_inclusion(id_pred, current_samples)
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



class UpdatePrealignedRobust:

    def __init__(self, server, conn, handle):
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

        # predict target user
        predicted_id = 10

        # accumulate samples - check for inconsistencies
        is_safe, samples = server.classifier.update_controller.accumulate_save_samples(embeddings)

        if is_safe:
            # add to data model
            server.classifier.data_controller.add_samples(user_id=user_id, new_samples=samples)
            # add to classifier training queue
            server.classifier.add_training_data(user_id, samples)
        else:
            # do prediction
            pass

        # return

        # submit data
        succ, conf = server.classifier.process_labeled_stream_data(user_id, embeddings, check_update=True)

        if succ == None:
            log.info('cls', "Update samples are unambiguously.")
            r.Error(server, conn, "Update samples are unambiguously.")
        elif succ == False:
            # update was not classified as the labeled class
            # force reidentification
            r.Reidentification(server, conn)
        else:
            r.UpdateFeedback(server, conn, int(conf*100))


# --------------- MISC


class CancelIdentification:

    def __init__(self, server, conn, handle):
        # receive tracking id
        tracking_id = server.receive_uint(conn)
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
        user_id_predicted = server.classifier.predict(embedding)

        # check if correct user
        if user_id_predicted is None:
            r.Error(server, conn, "Label could not be predicted - Face is unambiguous.")
            return
        elif user_id_predicted != user_id:
            # unknown user
            r.Error(server, conn, "The profile image does not come from the same person!")
            return

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
