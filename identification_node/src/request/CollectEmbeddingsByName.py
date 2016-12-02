#!/usr/bin/python
from src.response.Error import Error as ErrorResponse
from src.response.OK import OK as OKResponse


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
            ErrorResponse(server, conn, "No embeddings could be generated off the images")
            return

        # add to database
        server.user_db.add_embeddings(user_id, embeddings)

        OKResponse(server, conn)
