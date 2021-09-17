import numpy as np
import cv2


def preprocess_resnet18(image_file, norm=True):
    """
    Preprocessing of resnet18:
    Note that resnet18.pb is converted from resnet18.caffemodel, which originally uses bgr color.
    :param image_file:
    :return:
    """
    im = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError('No such image: {}'.format(image_file))
    try:
        im_dim = im.shape[2]
    except IndexError:
        im_dim = 1
    if im_dim == 4:
        bgr = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    elif im_dim == 1:
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        bgr = im
    image = np.asarray(bgr, dtype=np.float32)

    # resize
    h, w = image.shape[0], image.shape[1]
    new_height, new_width = 224, 224
    shortest_side = 256
    scale = float(shortest_side) / w if h > w else float(shortest_side) / h
    scaled_h = int(round(h * scale))
    scaled_w = int(round(w * scale))
    resized_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # central crop
    offset_height = int((scaled_h - new_height) / 2)
    offset_width = int((scaled_w - new_width) / 2)
    cropped_image = resized_image[offset_height:offset_height+new_height, offset_width:offset_width+new_width, :]

    # subtract by mean
    if norm:
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        mean = np.array([[[_B_MEAN, _G_MEAN, _R_MEAN]]])
        image = cropped_image - mean
    else:
        image = np.round(cropped_image).astype('uint8')
    return np.expand_dims(image, axis=0)


def image_preprocess(img_path, norm=True):
    return preprocess_resnet18(img_path, norm=norm)
