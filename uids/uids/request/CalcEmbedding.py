#!/usr/bin/python
from uids.response.Error import Error as ErrorResponse
from uids.response.Embedding import Embedding as EmbeddingResponse


class CalcEmbedding:

    def __init__(self, server, conn):

        # receive image size
        img_size = server.receive_uint(conn)
        # receive image
        user_face = server.receive_rgb_image(conn, img_size, img_size)
        # generate embedding
        embedding = server.embedding_gen.get_embedding(user_face)

        if embedding is None:
            ErrorResponse(server, conn, "Could not generate face embedding.")
            return

        EmbeddingResponse(server, conn, embedding)
