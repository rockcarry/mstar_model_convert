# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image

def get_image(img_path, resizeH=288, resizeW=352, resizeC=3, norm=True, meanB=104, meanG=117, meanR=123, std=1, nchw=False):
    img = np.array(Image.open(img_path).convert("RGB"))
    img_norm = cv2.resize(img, (resizeW, resizeH))

    if norm and (resizeC == 3):
        img_norm = img_norm.astype('float32')
        img_norm = (img_norm - [meanR, meanG, meanB]) / std

    else:
        pass

    if nchw:
        # NCHW
        img_norm = np.transpose(img_norm, axes=(2, 0, 1))

    return np.expand_dims(img_norm, 0)


def image_preprocess(img_path, norm=True):
    return get_image(img_path, norm=norm)
