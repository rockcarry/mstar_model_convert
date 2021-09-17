# -*- coding: utf-8 -*-

import cv2
import numpy as np

def get_image(img_path, resizeH=28, resizeW=28, norm=True, meanR=33.318, std=1, rgb=False, nchw=False):
    img = cv2.imread(img_path, flags=-1)
    if img is None:
        raise FileNotFoundError('No such image: {}'.format(img_path))
    try:
        img_dim = img.shape[2]
    except IndexError:
        img_dim = 1
    if img_dim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img_dim == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img_norm = cv2.resize(img, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)

    if norm:
        img_norm = (img_norm - meanR) / std
        img_norm = np.expand_dims(img_norm, axis=2) 
        dummy = np.zeros((28,28,2))
        img_norm = np.concatenate((img_norm,dummy),axis=2)
        img_norm = img_norm.astype('float32')
    else:
        img_norm = np.round(img_norm).astype('uint8')

    if rgb:
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)

    if nchw:
        # NCHW
        img_norm = np.transpose(img_norm.reshape(resizeW, resizeH, -1), axes=(2, 0, 1))

    return np.expand_dims(img_norm, 0)


def image_preprocess(img_path, norm=True):
    return get_image(img_path, norm=norm)
