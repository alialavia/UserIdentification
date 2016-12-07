from lib.EmbeddingGen import EmbeddingGen   # CNN embedding generator
import argparse
import os
from scipy import misc
import numpy as np
import time
import pickle

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))

def calc_embeddings(in_folder, gen, cleanup=False):

    path_in = os.path.join(fileDir, in_folder)
    embeddings = []
    print "--- starting to generate embeddings..."
    if path_in[-1:] is not "/":
        path_in = path_in + "/"

    start = time.time()

    tot_files = len(os.listdir(path_in))
    removed = 0
    file_nr = 1
    embeddings = []

    for index, file in enumerate(os.listdir(path_in)):
        if file.endswith(".jpg") or file.endswith(".png"):
            print "--- Processing file {}/{}".format(index+1, tot_files)
            image = misc.imread(path_in+file)
            embedding = gen.get_embedding(image)
            if embedding is None:
                print "--- could not generate face embedding"
                if cleanup is True:
                    os.remove(path_in+file)
                removed = removed + 1
                continue
            else:
                embeddings.append(embedding)
        else:
            continue
    print "--- useable: {}/{} images".format(tot_files-removed, tot_files)
    return embeddings

# ================================= #
#              Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', help="Image folder.", default="faces")
    parser.add_argument('--output', help="Image folder.", default="face_embeddings")
    parser.add_argument('--clean', dest='clean', action='store_true')
    parser.add_argument('--no-clean', dest='clean', action='store_false')
    parser.add_argument('--save_emb', dest='save_embeddings', action='store_true')
    parser.add_argument('--no-save_em', dest='save_embeddings', action='store_false')
    parser.set_defaults(save_embeddings=False)
    parser.set_defaults(clean=False)

    # parse arguments
    args = parser.parse_args()
    emb_gen = EmbeddingGen()

    # do calculations
    embeddings = calc_embeddings(args.img_folder, emb_gen, args.clean)
    if args.save_embeddings is True:
        filename = "{}.pkl".format(args.output)
        print("--- Saving face embeddings to '{}'".format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(embeddings,f)
            f.close()
