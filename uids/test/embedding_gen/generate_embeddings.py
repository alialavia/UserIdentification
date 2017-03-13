from uids.features.EmbeddingGenLightCNN import EmbeddingGen  # CNN embedding generator
import argparse
import os
from scipy import misc
import numpy as np
import time
import pickle
import csv


# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))

def calc_embeddings(in_folder, gen, cleanup=False, align=True, log=''):

    path_in = os.path.join(fileDir, in_folder)
    print "--- starting to generate embeddings..."
    if path_in[-1:] is not "/":
        path_in = path_in + "/"

    tot_files = 0
    removed = 0
    file_nr = 1
    embeddings = []
    start = time.time()
    nr_processed = 0

    if log != '':
        with open(os.path.join(path_in, log), 'rb') as csvfile:
            print "Opened log: {}".format(os.path.join(path_in, log))
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                tot_files += 1
                img_path = os.path.join(path_in, row[0])
                image = misc.imread(img_path)
                # swap to bgr format
                image = image[::-1,:,::-1]
                embedding = gen.get_embedding(image, align=align)
                if embedding is None:
                    print "--- could not generate face embedding: file {}".format(row[0])
                    if cleanup is True:
                        os.remove(img_path)
                    removed = removed + 1
                    continue
                else:
                    nr_processed = nr_processed + 1
                    print "--- Processed {} images".format(nr_processed)
                    embeddings.append(embedding)
    else:
        print "Scanning directory for images..."
        tot_files = len(os.listdir(path_in))
        for index, file in enumerate(os.listdir(path_in)):
            if file.endswith(".jpg") or file.endswith(".png"):
                print "--- Processing file {}/{}".format(index+1, tot_files)
                image = misc.imread(path_in+file)
                # swap to bgr format
                image = image[::-1,:,::-1]
                print "file {}".format(file)
                embedding = gen.get_embedding(image, align=align)
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
    parser.add_argument('--align', dest='align', action='store_true')
    parser.add_argument('--no-align', dest='align', action='store_false')
    parser.add_argument('--log', help="Log file ", default="")
    parser.set_defaults(save_embeddings=False)
    parser.set_defaults(clean=False)
    parser.set_defaults(recursive=False)
    parser.set_defaults(align=True)

    # parse arguments
    args = parser.parse_args()

    if args.output != "face_embeddings":
        args.save_embeddings = True

    emb_gen = EmbeddingGen()

    # do calculations
    embeddings = calc_embeddings(args.img_folder, emb_gen, args.clean, align=args.align, log=args.log)
    if args.save_embeddings is True:
        filename = "{}.pkl".format(args.output)
        print("--- Saving face embeddings to '{}'".format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)
            f.close()
