#!/usr/bin/env python2
import os
import time
import numpy as np
import caffe
from uids.utils.Logger import Logger as log
from skimage.transform import resize
from uids.utils.FaceAligner import FaceAligner

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models')	# path to the model directory
cnnModelDir = os.path.join(modelDir, 'light_cnn')


class EmbeddingGen:

    # settings
    size = 144      # see paper
    verbose = True
    networkModel = os.path.join(cnnModelDir, 'LightenedCNN_C.caffemodel')  # torch network model
    networkDef = os.path.join(cnnModelDir, "LightenedCNN_C_deploy_reduced.prototxt")

    grayscale = True    # model needs grayscale conversion - see paper
    neural_net = None       # caffe network
    aligner = None

    def __init__(self):
        start = time.time()

        log.info('cnn', "EmbeddingGenLightCNN: loading models...")
        # load face aligner
        #self.aligner = FaceAligner

        # load neural net
        caffe.set_mode_cpu()
        self.neural_net = caffe.Classifier(
                         self.networkDef, self.networkModel,  # "Data set image mean of [Channels x Height x Width] dimensions "
                                                              # "(numpy array). Set to '' for no mean subtraction."
                         image_dims=[self.size, self.size],  # input size of images
                         mean=None,
                         input_scale=1.0,
                         raw_scale=1.0,  # raw input is multiplied by this scale (caffe needs grayscale range [0-255])
                         channel_swap=None  # input format: BGR (BGR is caffe and OpenCV default)
                         )

        log.info('cnn', "EmbeddingGenLightCNN: model loading took {} seconds".format("%.3f" % (time.time() - start)))

    #  ----------- EMBEDDING GENERATION
    def get_embeddings(self, input_images, align=True):


        start = time.time()
        embeddings = self.neural_net.predict(input_images, False)

        # if self.verbose:
        #     print("--- Neural network forward pass took {} seconds.".format(time.time() - start))

        return np.array(embeddings)

    def get_embedding(self, img, align=True):
        """

        :param img: BGR image or grayscale image in range [0-255]
        :param align:
        :return:
        """

        # user_img = resize(user_img, (self.size, self.size),mode='nearest')
        # image = skimage.color.rgb2gray(image)

        # convert bgr to greyscale
        if self.grayscale and img.shape[2] != 1:
            # normalize image to range [0, 1]
            img = self.bgr2gray(img)
            #  size (H x W x 1) in grayscale
            img = img[:, :, np.newaxis]

        # resize
        img = self.resize_image(img, (self.size, self.size))

        # network processes in batch
        input_images = np.array([img])

        # generate embedding
        rep = self.neural_net.predict(input_images, False)
        return rep

    #  ----------- HELPERS

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.587, 0.114])

    def bgr2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.114, 0.587, 0.2989])

    def resize_image(self, im, new_dims, interp_order=1):
        """
        @ source: caffe io.py
        Resize an image array with interpolation.
        Parameters
        ----------
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        interp_order : interpolation order, default is linear.
        Returns
        -------
        im : resized ndarray with shape (new_dims[0], new_dims[1], K)
        """
        if im.shape[-1] == 1 or im.shape[-1] == 3:
            im_min, im_max = im.min(), im.max()
            if im_max > im_min:
                # skimage is fast but only understands {1,3} channel images
                # in [0, 1].
                im_std = (im - im_min) / (im_max - im_min)
                resized_std = resize(im_std, new_dims, order=interp_order)
                resized_im = resized_std * (im_max - im_min) + im_min
            else:
                # the image is a constant -- avoid divide by 0
                ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                               dtype=np.float32)
                ret.fill(im_min)
                return ret
        else:
            # ndimage interpolates anything but more slowly.
            scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
            resized_im = zoom(im, scale + (1,), order=interp_order)
        return resized_im.astype(np.float32)
