# -*- coding: utf-8 -*-

import cv2
import numpy as np


def get_image(img_path, resizeH=224, resizeW=224, resizeC=3, norm=True, meanB=104, meanG=117, meanR=123, std=1, rgb=False, nchw=False):
    img = cv2.imread(img_path, flags=-1)
    if img is None:
        raise FileNotFoundError('No such image: {}'.format(img_path))

    try:
        img_dim = img.shape[2]
    except IndexError:
        img_dim = 1
    if img_dim == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img_dim == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_float = img.astype('float32')
    img_norm = cv2.resize(img_float, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)

    if norm and (resizeC == 3):
        #mean_vedc = np.array([0.485, 0.456, 0.406])
        #stddev_vec = np.array([0.229, 0.224, 0.225])
        mean_vedc = np.array([123.68,116.28,103.53])
        stddev_vec = np.array([58.395,57.12,57.375])

        img_norm = (img_norm - [mean_vedc[0],mean_vedc[1],mean_vedc[2]]) / stddev_vec
        img_norm = img_norm.astype('float32')
        #img_norm = (img_norm - [meanB, meanG, meanR]) / std
        #img_norm = img_norm.astype('float32')
    elif norm and (resizeC == 1):
        img_norm = (img_norm - meanB) / std
        img_norm = img_norm.astype('float32')
    else:
        img_norm = np.round(img_norm).astype('uint8')

    if rgb:
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)

    if nchw:
        # NCHW
        img_norm = np.transpose(img_norm, axes=(2, 0, 1))

    return np.expand_dims(img_norm, 0)


def image_preprocess(img_path, norm=True):
    return get_image(img_path, norm=norm)
