import numpy as np
import cv2


def preprocess_resnet_v2_50(image_file, resizeH=299, resizeW=299, norm=True, central_crop=.875):
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

    h, w = rgb.shape[0], rgb.shape[1]
    h_cropped = int(h * central_crop)
    w_cropped = int(w * central_crop)
    h_start = (h - h_cropped) // 2
    h_end = h_cropped + h_start
    w_start = (w - w_cropped) // 2
    w_end = w_cropped + w_start
    image = rgb[h_start:h_end, w_start:w_end, :]

    resized_image = cv2.resize(image, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)
    if norm:
        # normalize to [-1,1]
        normalized_image = (2.0 / 255.0) * resized_image - 1.0
        normalized_image = normalized_image.astype('float32')
    else:
        normalized_image = np.round(resized_image).astype('uint8')
    return np.expand_dims(normalized_image, axis=0)


def image_preprocess(img_path, norm=True):
    return preprocess_resnet_v2_50(img_path, norm=norm)
