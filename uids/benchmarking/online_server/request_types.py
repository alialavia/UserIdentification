#!/usr/bin/python
import response_types as r
from uids.utils.Logger import Logger as log
import csv
import time
# config
from config import ROUTING
r.ROUTING = ROUTING


# --------------- IDENTIFICATION

class ImageReceival:

    def __init__(self, server, conn):

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)
        r.Pong(server, conn)
        

class FeatureGeneration:

    def __init__(self, server, conn):

        # receive images
        images = server.receive_image_batch_squared_same_size(conn)

        current_milli_time = lambda: int(round(time.time() * 1000))

        # generate embedding
        start = current_milli_time()
        embeddings = server.embedding_gen.get_embeddings(images, align=False)

        if not embeddings.any():
            r.Error(server, conn, "Could not generate face embeddings.")
            return

        timing = current_milli_time() - start
        filename = "timings - "+str(len(images))+" images.csv"
        with open(filename, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([timing])

        r.Pong(server, conn)


class Ping:

    def __init__(self, server, conn):
        r.Pong(server, conn)
