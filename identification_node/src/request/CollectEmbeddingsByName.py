#!/usr/bin/python
from src.response.Error import Error as ErrorResponse
from src.response.OK import OK as OKResponse


class CollectEmbeddingsByName:

    def __init__(self, server, conn):

        # receive user name size
        user_name_bytes = server.receive_int(conn)

        # receive user nice name
        user_nice_name = server.receive_message(conn, user_name_bytes)

        # receive image dimension
        img_dim = server.receive_int(conn)

        # receive batch size
        nr_images = server.receive_char(conn)

        print "--- Image batch received: size: " + str(nr_images) + " | image dimension: " + str(img_dim)

        # receive image batch
        images = []
        for x in range(0, nr_images):
            # receive image
            new_img = server.receive_rgb_image(conn, img_dim, img_dim)
            images.append(new_img)

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
