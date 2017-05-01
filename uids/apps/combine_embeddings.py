#!/usr/bin/env python2
import argparse
import os
import pickle
import numpy as np

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))

def load_data(filename):
    filename = "{}/{}".format(fileDir, filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embeddings = pickle.load(f)
            f.close()
        if not isinstance(embeddings, (np.ndarray, np.generic)):
            embeddings = np.array(embeddings)
        return embeddings
    else:
        print "File not found!"
    return None

# ------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('files', nargs='*', help = 'some ids')
    parser.add_argument('--output', help="pkl name", default="combined.pkl")
    args = parser.parse_args()

    # load all files

    combined = np.array(())

    print args.files
    for file in args.files:
        # load
        content = load_data(file)
        combined = np.concatenate((combined, content)) if combined.size else content

    # save
    print "Combined length: {}".format(len(combined))
    filename = args.output
    print("--- combined content to '{}'".format(filename))
    with open(filename, 'wb') as f:
        pickle.dump(combined, f)
        f.close()
