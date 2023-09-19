# -*- coding: utf-8 -*-

import cv2
import numpy as np


def get_image(img_path, resizeH=600, resizeW=800, resizeC=3, norm=False, meanB=104, meanG=117, meanR=123, std=1, rgb=True, nchw=False):
    img = cv2.imread(img_path, flags=-1)
    if img is None:
        raise FileNotFoundError('No such image: {}'.format(img_path))
    try:
        img_dim = img.shape[2]
    except IndexError:
        img_dim = 1
    if (img_dim == 3) and (resizeC == 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif (img_dim == 4) and (resizeC == 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif (img_dim == 4) and (resizeC == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif (img_dim == 1) and (resizeC == 3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_float = img.astype('float32')
    img_norm = cv2.resize(img_float, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)

    if not norm:
        img_norm = np.round(img_norm).astype('uint8')

    if rgb:
        if resizeC == 3:
            img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
        else:
            print('\033[31mImage channel is {}, no need to switch channels.\033[0m'.format(resizeC))

    if nchw:
        # NCHW
        img_norm = np.transpose(img_norm, axes=(2, 0, 1))
    return np.expand_dims(img_norm, 0)


def image_preprocess(img_path, norm=True):
    return get_image(img_path, norm=norm)
