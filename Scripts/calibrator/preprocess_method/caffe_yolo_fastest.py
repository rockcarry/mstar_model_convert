# -*- coding: utf-8 -*-

import cv2
import numpy as np

def get_image(img_path, resizeH=320, resizeW=320, resizeC=3, norm=True, meanB=0, meanG=0, meanR=0, std=255, rgb=True, nchw=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError('No such image: {}'.format(img_path))
    img_float = img.astype('float32')
    img_norm = cv2.resize(img_float, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)
    if norm:
        img_norm = (img_norm - [meanB, meanG, meanR]) / std
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
