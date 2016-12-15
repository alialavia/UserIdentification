from uids.online_learning.OCCTree import OneClassDetectorTree
import os
import pickle
import numpy as np

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models', 'embedding_samples')	# path to the model directory


def load_embeddings(filename):
    filename = "{}/{}".format(modelDir, filename)

    print filename
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        return np.array(embeddings)
    return None

# ================================= #
#              Main

if __name__ == '__main__':

    emb1 = load_embeddings("embeddings_elias.pkl")
    emb2 = load_embeddings("embeddings_matthias.pkl")
    emb3 = load_embeddings("embeddings_laia.pkl")
    emb_lfw = load_embeddings("embeddings_lfw.pkl")

    clf = OneClassDetectorTree(classifier='OCSVM')

    np.random.shuffle(emb1)
    np.random.shuffle(emb2)
    np.random.shuffle(emb3)
    np.random.shuffle(emb_lfw)

    split_set = np.array_split(emb1, 6)
    training_1 = split_set[0:3]
    test_1 = split_set[3:6]

    split_set = np.array_split(emb2, 6)
    training_2 = split_set[0:3]
    test_2 = split_set[3:6]

    split_lfw = np.array_split(emb_lfw, 6)

    for i in range(3):
        if i==0:
            # add test class
            if not clf.init_classifier(1, training_1[i]):
                print "--- initialization failed"
            if not clf.init_classifier(2, training_2[i]):
                print "--- initialization failed"
        else:
            print "----PREDICTION: SET 1----------"
            print clf.predict_class(test_1[i])
            print "-------------------------------"

            clf.process_labeled_stream_data(1, training_1[i])
            clf.process_labeled_stream_data(2, training_2[i])
            print "----PREDICTION: SET 1----------"
            print clf.predict_class(test_1[i])
            print "-------------------------------"
            print "----PREDICTION: SET 2----------"
            print clf.predict_class(test_2[i])
            print "-------------------------------"
            print "----PREDICTION: SET LWF----------"
            print clf.predict_class(split_lfw[i])
            print "-------------------------------"

    while True:
        pass