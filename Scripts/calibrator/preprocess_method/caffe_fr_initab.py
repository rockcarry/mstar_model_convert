import numpy as np
import cv2


def preprocess_caffe_fr(image_file, norm):
    """
    Preprocessing of caffe_fr:
    normalize [-1,1]
    :return:
    """
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


    if norm:
        # normalize to [-1,1]
        normalized_image = (2.0 / 255.0) * rgb - 1.0
    else:
        normalized_image = np.round(rgb).astype('uint8')

    return np.expand_dims(normalized_image, 0)


def image_preprocess(img_path, norm=True):
    return preprocess_caffe_fr(img_path, norm=norm)

