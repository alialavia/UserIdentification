from uids.features.EmbeddingGen import EmbeddingGen   # CNN embedding generator
import argparse
import os
from scipy import misc
import numpy as np
import time
import pickle
import csv

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))

def calc_embeddings_recursive(in_folder, gen, output, batch_size = 100, cleanup = False, align=True):

    path_in = os.path.join(fileDir, in_folder)
    print "--- starting to generate embeddings..."
    if path_in[-1:] is not "/":
        path_in = path_in + "/"

    removed = 0
    embeddings = []
    start = time.time()
    batch_id = 1
    nr_processed = 0

    # generate embeddings for all files in folder
    for root, subdirs, files in os.walk(path_in):
        #print('--\nroot = ' + root)

        tot_files = len(files)

        for index, filename in enumerate(files):
            file = os.path.join(root, filename)

            if file.endswith(".jpg") or file.endswith(".png"):

                #print "--- Processing file {}/{}".format(index + 1, tot_files)
                # load image in RGB order
                image = misc.imread(file)
                embedding = gen.get_embedding(image, align=align)
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


def calc_embeddings(in_folder, gen, cleanup=False, align=True, log='', save_pose=True):

    path_in = os.path.join(fileDir, in_folder)
    print "--- starting to generate embeddings..."
    if path_in[-1:] is not "/":
        path_in = path_in + "/"

    tot_files = 0
    removed = 0
    file_nr = 1
    embeddings = []
    pose = []
    start = time.time()
    nr_processed = 0
    picture_names = []

    if log != '':
        with open(os.path.join(path_in, log), 'rb') as csvfile:
            print "Opended log: {}".format(os.path.join(path_in, log))

            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            reader = csv.reader(csvfile, dialect)

            # spamreader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                tot_files += 1
                img_path = os.path.join(path_in, row[0])

                if not os.path.exists(img_path):
                    print "--- File {} does not exists. Skipping...".format(row[0])
                    continue
                # load image in RGB order
                image = misc.imread(img_path)
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
                    picture_names.append(row[0])
                    # save pose
                    if save_pose:
                        pose.append([int(row[1]), int(row[2]), int(row[3])])
    else:
        print "Scanning directory for images..."
        tot_files = len(os.listdir(path_in))
        for index, file in enumerate(os.listdir(path_in)):
            if file.endswith(".jpg") or file.endswith(".png"):
                print "--- Processing file {}/{}".format(index+1, tot_files)
                # load image in RGB order
                image = misc.imread(path_in+file)
                print "file {}".format(file)
                embedding = gen.get_embedding(image, align=align)
                if embedding is None:
                    print "--- could not generate face embedding"
                    if cleanup is True:
                        os.remove(path_in+file)
                    removed = removed + 1
                    continue
                else:
                    nr_processed = nr_processed + 1
                    embeddings.append(embedding)
                    picture_names.append(file)
            else:
                continue

    print "--- embedding calculation took {} seconds".format(time.time()-start)
    print "--- useable: {}/{} images".format(nr_processed, tot_files)
    return embeddings, pose, picture_names

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
    parser.add_argument('--save_pose', dest='save_pose', action='store_true')
    parser.add_argument('--no-save_pose', dest='save_pose', action='store_false')
    parser.add_argument('--log', help="Log file ", default="")
    parser.set_defaults(save_embeddings=False)
    parser.set_defaults(clean=False)
    parser.set_defaults(recursive=False)
    parser.set_defaults(align=True)
    parser.set_defaults(save_pose=False)

    # parse arguments
    args = parser.parse_args()

    if args.output != "face_embeddings":
        args.save_embeddings = True

    if args.save_pose is True and args.log == '':
        print "Please specify a --log if you want to extract the face poses"

    emb_gen = EmbeddingGen()

    # do calculations
    if args.recursive:
        embeddings = calc_embeddings_recursive(args.img_folder, emb_gen, args.output, args.batch_size, args.clean, align=args.align)
    else:
        embeddings, pose, picture_names = calc_embeddings(args.img_folder, emb_gen, args.clean, align=args.align, log=args.log, save_pose=args.save_pose)
        if args.save_embeddings is True:
            filename = "{}.pkl".format(args.output)
            print("--- Saving face embeddings to '{}'".format(filename))
            with open(filename, 'wb') as f:
                pickle.dump(embeddings, f)
                f.close()
            with open("{}_image_names.pkl".format(args.output), 'wb') as f:
                pickle.dump(picture_names, f)
                f.close()

            if len(pose) > 0:
                filename = "{}_poses.pkl".format(args.output)
                print("--- Saving face poses to '{}'".format(filename))
                with open(filename, 'wb') as f:
                    pickle.dump(pose, f)
                    f.close()
