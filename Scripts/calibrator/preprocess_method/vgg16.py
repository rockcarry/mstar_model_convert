import numpy as np
import cv2


def preprocess_vgg_16(image_file, norm=True, resize_strategy='cv'):
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
    h, w = rgb.shape[0], rgb.shape[1]
    new_height, new_width = 224, 224
    shortest_side = 256
    scale = float(shortest_side) / w if h > w else float(shortest_side) / h
    scaled_h = int(round(h * scale))
    scaled_w = int(round(w * scale))

    image = np.asarray(rgb, dtype=np.float32)
    if resize_strategy == 'tf':
        import tensorflow as tf
        image = np.expand_dims(image, 0)
        tf_resized_image = tf.compat.v1.image.resize_bilinear(image, (scaled_w, scaled_h))
        with tf.Session():
            resized_image = tf_resized_image.eval()
    else:
        cv_resized_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        resized_image = cv_resized_image

    # central crop
    offset_height = int((scaled_h - new_height) / 2)
    offset_width = int((scaled_w - new_width) / 2)

    cropped_image = resized_image[offset_height:offset_height+new_height, offset_width:offset_width+new_width, :]

    if norm:
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        mean = np.array([[[_R_MEAN, _G_MEAN, _B_MEAN]]])
        image = cropped_image - mean
    else:
        image = np.round(cropped_image).astype('uint8')

    return np.expand_dims(image, axis=0)


def image_preprocess(img_path, norm=True):
    return preprocess_vgg_16(img_path, norm=norm)
