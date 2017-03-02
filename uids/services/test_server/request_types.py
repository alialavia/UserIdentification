#!/usr/bin/python
import response_types as r
from uids.utils.Logger import Logger as log
# config
from config import ROUTING
r.ROUTING = ROUTING

class Update:

    def __init__(self, server, conn):
        # receive user id
        user_id = server.receive_uint(conn)

        log.info('server', 'User Update for ID {}'.format(user_id))

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
        succ, conf = server.classifier.process_labeled_stream_data(user_id, embeddings, check_update=False)

        if succ is None:
            r.Error(server, conn, "Update samples are unambiguously.")
        else:
            r.UpdateFeedback(server, conn, int(conf * 100))


class Ping:

    def __init__(self, server, conn):
        r.Pong(server, conn)
