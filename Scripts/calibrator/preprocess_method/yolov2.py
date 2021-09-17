import cv2
import numpy as np


def preprocess_yolov2(image_file, norm=True, use_simple_preprocessing=True, use_lance_preprocess=True):
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
    if use_simple_preprocessing:
        new_height, new_width = 416, 416
        if use_lance_preprocess:
            img = cv2.resize(rgb, (new_width, new_height)).astype(np.float32)
            if norm:
                preprocessed = img / 255.0
            else:
                preprocessed = np.round(img).astype('uint8')
        else:
            image = rgb / 255.0
            preprocessed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    else:
        image = rgb / 255.0
        h, w = image.shape[0], image.shape[1]
        new_height, new_width = 416, 416
        scale = min(1. * new_height / h, 1. * new_width / w)
        scaled_h = int(round(h * scale))
        scaled_w = int(round(w * scale))

        preprocessed = np.full((new_height, new_width, 3), fill_value=0.5)
        resized_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        xmin = int((preprocessed.shape[0] - resized_image.shape[0]) / 2)
        xmax = resized_image.shape[0] + xmin
        ymin = int((preprocessed.shape[1] - resized_image.shape[1]) / 2)
        ymax = resized_image.shape[1] + ymin
        preprocessed[xmin:xmax, ymin:ymax, :] = resized_image[:, :, :]

    return np.expand_dims(preprocessed, axis=0)


def image_preprocess(img_path, norm=True):
    return preprocess_yolov2(img_path, norm=norm)
