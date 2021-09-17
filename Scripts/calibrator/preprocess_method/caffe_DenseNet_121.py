# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize

def resize_image(im, new_dims, interp_order=1):
    """
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
            resized_std = resize(im_std, new_dims, order=interp_order, mode='constant')
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

def get_image(img_path, resizeH=224, resizeW=224, resizeC=3, norm=True, meanB=103.94, meanG=116.78, meanR=123.68, std=58.82, rgb=False, nchw=False):

    im = cv2.imread(img_path)
    if im is None:
        raise FileNotFoundError('No such image: {}'.format(img_path))
    im = im / 255.
    h, w, _ = im.shape
    if h < w:
        off = (w - h) // 2
        im = im[:, off:off + h]
    else:
        off = (h - w) // 2
        im = im[off:off + h, :]
    im = resize_image(im, [256, 256])
    im = im[16:240, 16:240, :]
    im = im * 255
    if norm:
        _mean = np.array([meanB, meanG, meanR]).reshape([1,1,3])
        _im = im - _mean
        _im = _im * 0.017
        _im = _im.astype(np.float32)
    else:
        _im = np.uint8(im)
    return np.expand_dims(_im, 0)

def image_preprocess(img_path, norm=True):
    return get_image(img_path, norm=norm)
