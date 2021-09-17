import numpy as np
import cv2
import math

def preprocess_nasnet(image_file, resizeH=224, resizeW=224, norm=True, central_crop=.875):


    """
    Pre-processing of resnet_v2_50 (same as inception_v3):
    central_crop -> resize -> normalize
    """

    im = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError('No such image: {}'.format(image_file))
    im_dim = np.ndim(im)
    if im_dim == 3:
        if im.shape[-1] == 3:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            assert im.shape[-1] == 4
            rgb = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    _max = rgb.max()
    rgb = rgb/_max
    h, w = rgb.shape[0], rgb.shape[1]
    h_cropped = math.floor((h - h * central_crop) / 2.0)
    w_cropped = math.floor((w - w * central_crop) / 2.0)
    h_start = h_cropped
    h_end = h - h_cropped
    w_start = w_cropped
    w_end = w - w_cropped
    image = rgb[h_start:h_end, w_start:w_end, :]

    resized_image = cv2.resize(image, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)

    if norm:
        # normalize to [-1,1]
        normalized_image = resized_image - 0.5
        normalized_image = normalized_image * 2
    else:
        normalized_image = np.array(resized_image * _max,dtype='uint8')
    return np.expand_dims(normalized_image, axis=0)


def image_preprocess(img_path, norm=True):
    return preprocess_nasnet(img_path, norm=norm)
