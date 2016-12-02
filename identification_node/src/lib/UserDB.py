#!/usr/bin/env python

import time
import os
import pickle
import numpy as np

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
DBDir = os.path.join(fileDir, '../..', 'db')  # path to the database directory


class UserDB:

    version_name = "v1-0"
    id_increment = 1        # user id increment
    user_list = {}          # user ids to nice name

    # user associated data
    user_embeddings = {}    # raw CNN embeddings

    def __init__(self):
        start = time.time()
        print "--- loading user database..."
        if self.load() is False:
            print "    Not database exists yet. Initializing new one..."
        print("--- database initialization took {} seconds".format(time.time() - start))

    #  ----------- DESCRIPTOR TOOLS
    def add_embeddings(self, user_id, new_embeddings):
        """embeddings: array of embeddings"""
        if user_id not in self.user_embeddings:
            # initialize
            self.user_embeddings[user_id] = new_embeddings
        else:
            # append
            self.user_embeddings[user_id] = np.concatenate((self.user_embeddings[user_id], new_embeddings))

    def get_labeled_embeddings(self):
        embeddings_accumulated = np.array([])
        labels = []
        # label encoder id: np.int64()
        for user_id, user_embeddings in self.user_embeddings.iteritems():
            labels = np.append(labels, np.repeat(user_id, len(user_embeddings)))
            embeddings_accumulated = np.concatenate((embeddings_accumulated, user_embeddings)) if embeddings_accumulated.size else np.array(user_embeddings)
        return embeddings_accumulated, labels

    #  ----------- USER TOOLS
    def get_name_from_id(self, user_id):
        if user_id in self.user_list:
            return self.user_list[user_id]
        else:
            return None

    def get_id_from_name(self, search_name):
        for user_id, name in self.user_list.iteritems():
            if name == search_name:
                return user_id
        return None

    def create_new_user(self, nice_name):
        # generate unique id
        new_id = self.id_increment
        self.id_increment += 1  # increment user id
        # save nice name
        self.user_list[int(new_id)] = nice_name
        return new_id

    # ----------- STORAGE
    def load(self):
        filename = "{}/userdb_{}.pkl".format(DBDir, self.version_name)
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                (
                    self.version_name,
                    self.id_increment,
                    self.user_list,
                    self.user_embeddings
                ) = pickle.load(f)
            return True
        return False

    def save(self):
        filename = "{}/userdb_{}.pkl".format(DBDir, self.version_name)
        print("--- Saving database to '{}'".format(filename))
        with open(filename, 'wb') as f:
            pickle.dump((
                self.version_name,
                self.id_increment,
                self.user_list,
                self.user_embeddings
            ), f)

    # ----------- DISPLAY
    def print_users(self):
        if len(self.user_list) == 0:
            print "--- No users found in the database"

        print "--- Current users:"
        for user_id, name in self.user_list.iteritems():
            print "     User: ID(" + str(user_id) + "): " + name

    def print_embedding_status(self):
        print "--- Current embeddings:"
        for user_id, embeddings in self.user_embeddings.iteritems():
            print "     User" + str(user_id) + ": " + str(len(embeddings)) + " representations"