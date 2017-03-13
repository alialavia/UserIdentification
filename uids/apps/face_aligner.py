import argparse
import os
from scipy import misc
import numpy as np
import time
import pickle
import openface
from scipy.misc import imsave

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')


dlib_aligner = openface.AlignDlib(dlibModelDir + "/shape_predictor_68_face_landmarks.dat")

def align_face(image, output_size=96, skip_multi=False):
    outRgb = dlib_aligner.align(output_size, image,
                                     landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE,
                                     skipMulti=skip_multi)
    # out Rgb might be none
    return outRgb

def align_images_in_folder(in_folder, cleanup=False, save=False, replace=False):

    path_in = os.path.join(fileDir, in_folder)
    print "--- starting to generate embeddings..."
    if path_in[-1:] is not "/":
        path_in = path_in + "/"

    tot_files = len(os.listdir(path_in))
    removed = 0
    file_nr = 1
    start = time.time()
    aligned_faces = []

    for index, file in enumerate(os.listdir(path_in)):
        if file.endswith(".jpg") or file.endswith(".png"):
            print "--- Processing file {}/{}".format(index+1, tot_files)
            image = misc.imread(path_in+file)
            # print "file {}".format(file)

            aligned = align_face(image)
            if aligned is None:
                print "--- could not align face (img: {})".format(file)
                if cleanup is True:
                    os.remove(path_in+file)
                removed = removed + 1
                continue
            else:
                if save is True:
                    if replace is True:
                        imsave(path_in + file, aligned)
                    else:
                        imsave(path_in+'aligned_'+file, aligned)
                aligned_faces.append(aligned)
        else:
            continue

    print "--- alignment took {} seconds".format(time.time()-start)
    print "--- useable: {}/{} images".format(tot_files-removed, tot_files)
    return aligned_faces

def write_png(buf, width, height):
    """ buf: must be bytes or a bytearray in Python3.x,
        a regular string in Python2.x.
    """
    import zlib, struct

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                        for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

def saveAsPNG(array, filename):
    import struct
    if any([len(row) != len(array[0]) for row in array]):
        raise ValueError, "Array should have elements of equal size"

                                #First row becomes top row of image.
    flat = []; map(flat.extend, reversed(array))
                                 #Big-endian, unsigned 32-byte integer.
    buf = b''.join([struct.pack('>I', ((0xffFFff & i32)<<8)|(i32>>24) )
                    for i32 in flat])   #Rotate from ARGB to RGBA.

    data = write_png(buf, len(array[0]), len(array))
    f = open(filename, 'wb')
    f.write(data)
    f.close()


# ================================= #
#              Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--imf', help="Image folder.", default="matthias_big")
    parser.add_argument('--clean', dest='clean', action='store_true')
    parser.add_argument('--no-clean', dest='clean', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.add_argument('--replace', dest='replace', action='store_true')
    parser.add_argument('--no-replace', dest='replace', action='store_false')
    parser.set_defaults(clean=False)
    parser.set_defaults(save=True)
    parser.set_defaults(replace=False)

    # parse arguments
    args = parser.parse_args()

    aligned_faces = align_images_in_folder(args.imf, cleanup=args.clean, save=args.save, replace=args.replace)