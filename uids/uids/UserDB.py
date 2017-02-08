#!/usr/bin/env python
import time
import os
import pickle
import numpy as np
from uids.utils.Logger import Logger as log

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
DBDir = os.path.join(fileDir, '..', 'db')  # path to the database directory


class UserDB:

    # TODO: implement intelligent storage:
    # -- save to hd if not used in a while (Memory-map files)
    # -- define memory limit
    # -- add data reduction method (clean out data set, time windows)

    version_name = "v1-0"
    id_increment = 1        # user id increment
    __user_list = {}          # user ids to nice name

    # user associated data
    # Todo: remove this - data is only kept in classifier cluster
    __class_samples = {}    # raw CNN embeddings
    # profile pictures
    __profile_pictures = {}

    def __init__(self):
        start = time.time()
        if self.load() is False:
            log.info('db', "No database exists yet. Created new one...")
        else:
            log.info('db', "Database initialization took {} seconds".format( "%.5f" % (time.time() - start)))

    # ------------ Profile pictures

    def set_profile_picture(self, user_id, picture):
        self.__profile_pictures[user_id] = picture

    def get_profile_picture(self, user_id):
        if user_id in self.__profile_pictures:
            return self.__profile_pictures[user_id]
        else:
            return None

    #  ----------- DESCRIPTOR TOOLS
    def add_samples(self, user_id, new_samples):
        """embeddings: array of embeddings"""
        if user_id not in self.__class_samples:
            # initialize
            self.__class_samples[user_id] = new_samples
        else:
            # append
            self.__class_samples[user_id] = np.concatenate((self.__class_samples[user_id], new_samples))

    def get_class_samples(self, class_id):
        if class_id in self.__class_samples:
            return self.__class_samples[class_id]
        else:
            return None

    def get_labeled_samples(self):
        embeddings_accumulated = np.array([])
        labels = []
        # label encoder id: np.int64()
        for user_id, user_embeddings in self.__class_samples.iteritems():
            labels = np.append(labels, np.repeat(user_id, len(user_embeddings)))
            embeddings_accumulated = np.concatenate((embeddings_accumulated, user_embeddings)) if embeddings_accumulated.size else np.array(user_embeddings)
        return embeddings_accumulated, labels

    #  ----------- USER TOOLS
    def get_name_from_id(self, user_id):
        if user_id in self.__user_list:
            return self.__user_list[user_id]
        else:
            return None

    def get_id_from_name(self, search_name):
        for user_id, name in self.__user_list.iteritems():
            if name == search_name:
                return user_id
        return None

    def create_new_user(self, nice_name):
        # generate unique id
        new_id = self.id_increment
        self.id_increment += 1  # increment user id
        # save nice name
        self.__user_list[int(new_id)] = nice_name
        return new_id

    # ----------- STORAGE
    def load(self):
        filename = "{}/userdb_{}.pkl".format(DBDir, self.version_name)
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                (
                    self.version_name,
                    self.id_increment,
                    self.__user_list,
                    self.__class_samples,
                    self.__profile_pictures
                ) = pickle.load(f)
                f.close()
            return True
        return False

    def save(self):
        filename = "{}/userdb_{}.pkl".format(DBDir, self.version_name)
        log.info('db', "Saving database to '{}'".format(filename))
        with open(filename, 'wb') as f:
            pickle.dump((
                self.version_name,
                self.id_increment,
                self.__user_list,
                self.__class_samples,
                self.__profile_pictures
            ), f)
            f.close()

    # ----------- DISPLAY
    def print_users(self):
        if len(self.__user_list) == 0:
            log.info('db', "No users found in the database")
        else:
            log.info('db', "Current Users:")
            for user_id, name in self.__user_list.iteritems():
                log.info('db', "     {} [ID] - {} [username]".format(user_id, name))

    def print_embedding_status(self):
        log.info('db', "Current embeddings:")
        for user_id, embeddings in self.__class_samples.iteritems():
            log.info('db', "     User" + str(user_id) + ": " + str(len(embeddings)) + " representations")
