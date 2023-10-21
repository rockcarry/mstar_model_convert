import numpy as np
import cv2


def preprocess_mobilenet_v1(image_file, norm=True, resize_strategy='cv', central_crop=.875):
    """
    Preprocessing of mobilenet_v1:
    central_crop -> resize -> normalize
    :param image_file:
    :param resize_strategy:
    :param central_crop:
    :return:
    """
    im = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
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

    # 1. central crop
    h, w = rgb.shape[0], rgb.shape[1]
    h_cropped = int(h * central_crop)
    w_cropped = int(w * central_crop)
    h_start = (h - h_cropped) // 2
    h_end = h_cropped + h_start
    w_start = (w - w_cropped) // 2
    w_end = w_cropped + w_start
    image = rgb[h_start:h_end, w_start:w_end, :]

    # 2. resize to target shape
    (scaled_w, scaled_h) = (160, 160)
    if resize_strategy == 'tf':
        import tensorflow as tf
        image = np.expand_dims(image, 0)
        tf_resized_image = tf.compat.v1.image.resize_bilinear(image, (scaled_w, scaled_h))
        with tf.Session():
            resized_image = tf_resized_image.eval()
    else:
        cv_resized_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        resized_image = np.expand_dims(cv_resized_image, 0)

    if norm:
        # normalize to [-1,1]
        normalized_image = (1.0 / 255.0) * resized_image
        normalized_image = normalized_image.astype('float32')
    else:
        normalized_image = resized_image

    return normalized_image


def image_preprocess(img_path, norm=True):
    return preprocess_mobilenet_v1(img_path, norm=norm)