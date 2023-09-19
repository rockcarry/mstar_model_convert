import numpy as np
import cv2


def get_image(img_path, resizeH=224, resizeW=224, resizeC=3, norm=True, meanB=117, meanG=117, meanR=117, std=1, rgb=False, nchw=False):
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError('No such image: {}'.format(img_path))
    try:
        im_dim = im.shape[2]
    except IndexError:
        im_dim = 1
    if im_dim == 3:
        bgr = im
    elif im_dim == 4:
        bgr = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    else:
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    image = bgr.astype(np.float32)

    h, w = image.shape[0], image.shape[1]
    shortest_side = max(resizeH, resizeW)
    scale = float(shortest_side) / w if h > w else float(shortest_side) / h
    scaled_h = int(round(h * scale))
    scaled_w = int(round(w * scale))
    resized_image = cv2.resize(image, (scaled_w, scaled_h), cv2.INTER_LINEAR)

    # # # central crop
    offset_height = int((scaled_h - resizeH) / 2)
    offset_width = int((scaled_w - resizeW) / 2)
    cropped_image = resized_image[offset_height:offset_height+resizeH, offset_width:offset_width+resizeW, :]

    if norm is True:
        normalized_image = (cropped_image - [meanB, meanG, meanR]) / std
    else:
        normalized_image = np.round(cropped_image).astype('uint8')

    if rgb is True:
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)

    if nchw is True:
        normalized_image = np.transpose(normalized_image, axes=(2, 0, 1))
    return np.expand_dims(normalized_image, 0)


def image_preprocess(img_path, norm=True):
    return get_image(img_path, norm=norm)
