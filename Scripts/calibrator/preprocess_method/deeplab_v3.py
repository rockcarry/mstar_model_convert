# -*- coding: utf-8 -*-

import cv2
import numpy as np


def preprocess_deeplab_v3(image_file, norm=True):
    im = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError('No such image: {}'.format(image_file))
    try:
        im_dim = im.shape[2]
    except IndexError:
        im_dim = 1
    if im_dim == 3:
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif im_dim == 4:
        rgb = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    image = rgb.astype(np.float32)

    height, width = image.shape[0], image.shape[1]

    resize_ratio = 513.0 / max(height, width)
    image = cv2.resize(
        rgb, (int(resize_ratio*width), int(resize_ratio*height)), interpolation=cv2.INTER_LINEAR)
    resized_height, resized_width = image.shape[0], image.shape[1]

    target_height = resized_height + max(513 - resized_height, 0)
    target_width = resized_width + max(513 - resized_width, 0)
    padded_value = np.reshape(np.array([127.5, 127.5, 127.5]), newshape=[1, 1, 3])
    padded_image = np.zeros(shape=[target_height, target_width, 3], dtype=np.float32)
    if target_height > resized_height:
        padded_image[:(target_height-resized_height), :, :] = padded_value
        padded_image[(target_height-resized_height):, (target_width-resized_width):, :] = image[:, :, :]
    if target_width > resized_width:
        padded_image[:, :(target_width-resized_width), :] = padded_value
        padded_image[(target_height-resized_height):, (target_width-resized_width):, :] = image[:, :, :]

    expanded_image = np.expand_dims(padded_image.astype(np.uint8), axis=0)
    if norm:
        # normalize to [-1,1]
        normalized_image = (2.0 / 255.0) * expanded_image - 1.0
    else:
        normalized_image = np.round(expanded_image).astype('uint8')
    return normalized_image


def image_preprocess(img_path, norm=False):
    return preprocess_deeplab_v3(img_path, norm=norm)
