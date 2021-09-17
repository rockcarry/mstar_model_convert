# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pdb
from PIL import Image
def get_image(img_path, resizeH=448, resizeW=768, resizeC=3, norm=True, meanB=0, meanG=0, meanR=0, std=255, rgb=False, nchw=False):
    img = cv2.imread(img_path, flags=-1)
    #image = Image.open(img_path)
    try:
        img_dim = img.shape[2]
    except IndexError:
        img_dim = 1
    if img_dim == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img_dim == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    img_float = img.astype('float32')
    #img_float = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = cv2.resize(img_float, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)
    if norm:
      #img_norm = (img_norm - [128, 128, 128]) / std
      img_norm = (img_norm - [meanB, meanG, meanR]) / std
      img_norm = img_norm.astype('float32')
    if rgb:
      img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
    #img_norm = np.load("input_data.npy")
    #img_norm = np.transpose(img_norm,axes=(0,2,3,1))
    img_norm = np.reshape(img_norm,(1,448,768,3))
    #pdb.set_trace()
    return img_norm


def image_preprocess(img_path, norm=True):
    return get_image(img_path, norm=norm)
