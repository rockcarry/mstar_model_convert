import numpy as np
import cv2
import pdb

def preprocess_shufflenet_v2(image_file, norm=True, resize_strategy='cv'):
    """
    Preprocessing of shufflenet:
    use bgr as input
    :param image_file:
    :param resize_strategy:
    :param central_crop:
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
        bgr = im
    elif im_dim == 4:
        bgr = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    else:
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    image = bgr.astype(np.float32)
    # im = cv2.imread(image_file, cv2.IMREAD_COLOR)
    # image = im.astype(np.float32)

    h, w = image.shape[0], image.shape[1]
    new_height, new_width = 224, 224
    shortest_side = 256
    scale = float(shortest_side) / w if h > w else float(shortest_side) / h
    scaled_h = int(round(h * scale))
    scaled_w = int(round(w * scale))
    resized_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # central crop
    offset_height = int((scaled_h - new_height) / 2)
    offset_width = int((scaled_w - new_width) / 2)
    cropped_image = resized_image[offset_height:offset_height+new_height, offset_width:offset_width+new_width, :]
    #pdb.set_trace()
    if norm:
        mean = np.array([255*0.4059999883174896, 255*0.4560000002384186, 255*0.48500001430511475],dtype=float)
        scale_div = np.array([255*0.22499999403953552, 255*0.2240000069141388, 255*0.2290000021457672],dtype=float)

        cropped_image = cropped_image - mean
        cropped_image = cropped_image/scale_div
    else:
        cropped_image = np.round(cropped_image).astype('uint8')

    return np.expand_dims(cropped_image, axis=0)


def image_preprocess(img_path, norm=True):
    return preprocess_shufflenet_v2(img_path, norm=norm)
