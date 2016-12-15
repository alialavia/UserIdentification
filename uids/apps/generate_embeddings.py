from uids.features.EmbeddingGen import EmbeddingGen   # CNN embedding generator
import argparse
import os
from scipy import misc
import numpy as np
import time
import pickle

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))

def calc_embeddings_recursive(in_folder, gen, output, batch_size = 100, cleanup = False):

    path_in = os.path.join(fileDir, in_folder)
    print "--- starting to generate embeddings..."
    if path_in[-1:] is not "/":
        path_in = path_in + "/"

    removed = 0
    embeddings = []
    start = time.time()
    batch_id = 1
    nr_processed = 0

    for root, subdirs, files in os.walk(path_in):
        #print('--\nroot = ' + root)

        tot_files = len(files)

        for index, filename in enumerate(files):
            file = os.path.join(root, filename)

            if file.endswith(".jpg") or file.endswith(".png"):

                #print "--- Processing file {}/{}".format(index + 1, tot_files)
                image = misc.imread(file)
                embedding = gen.get_embedding(image)
                if embedding is None:
                    print "--- could not generate face embedding"
                    if cleanup is True:
                        os.remove(path_in + file)
                    removed = removed + 1
                    continue
                else:
                    nr_processed = nr_processed + 1
                    print "--- Processed {} images".format(nr_processed)
                    embeddings.append(embedding)
            else:
                continue

            # save batch
            if len(embeddings) == batch_size:
                out = "{}_{}.pkl".format(output, batch_id)
                print("--- Saving face embeddings to '{}'".format(out))
                with open(out, 'wb') as f:
                    pickle.dump(embeddings, f)
                    batch_id = batch_id + 1
                    # clear list
                    embeddings[:] = []
                    f.close()

    if len(embeddings) > 0:
        # save as one file
        out = "{}_{}.pkl".format(output, batch_id)
        print("--- Saving face embeddings to '{}'".format(out))
        with open(out, 'wb') as f:
            pickle.dump(embeddings, f)
            batch_id = batch_id + 1
            # clear list
            embeddings[:] = []
            f.close()

    print "--- embedding calculation took {} seconds".format(time.time()-start)
    print "--- useable: {}/{} images".format(tot_files - removed, tot_files)
    return embeddings

def calc_embeddings(in_folder, gen, cleanup=False):

    path_in = os.path.join(fileDir, in_folder)
    print "--- starting to generate embeddings..."
    if path_in[-1:] is not "/":
        path_in = path_in + "/"

    tot_files = len(os.listdir(path_in))
    removed = 0
    file_nr = 1
    embeddings = []
    start = time.time()

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

    print "--- embedding calculation took {} seconds".format(time.time()-start)
    print "--- useable: {}/{} images".format(tot_files-removed, tot_files)
    return embeddings

# ================================= #
#              Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', help="Image folder.", default="faces")
    parser.add_argument('--output', help="output filename", default="face_embeddings")
    parser.add_argument('--batch_size', type=int, help="Batch storage size for recursive processing.", default=300)
    parser.add_argument('--clean', dest='clean', action='store_true')
    parser.add_argument('--no-clean', dest='clean', action='store_false')
    parser.add_argument('--save_emb', dest='save_embeddings', action='store_true')
    parser.add_argument('--no-save_emb', dest='save_embeddings', action='store_false')
    parser.add_argument('--recursive', dest='recursive', action='store_true')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false')
    parser.set_defaults(save_embeddings=False)
    parser.set_defaults(clean=False)
    parser.set_defaults(recursive=False)

    # parse arguments
    args = parser.parse_args()

    emb_gen = EmbeddingGen()

    # do calculations
    if args.recursive:
        embeddings = calc_embeddings_recursive(args.img_folder, emb_gen, args.output, args.batch_size, args.clean)
    else:
        embeddings = calc_embeddings(args.img_folder, emb_gen, args.clean)
        if args.save_embeddings is True:
            filename = "{}.pkl".format(args.output)
            print("--- Saving face embeddings to '{}'".format(filename))
            with open(filename, 'wb') as f:
                pickle.dump(embeddings,f)
                f.close()
