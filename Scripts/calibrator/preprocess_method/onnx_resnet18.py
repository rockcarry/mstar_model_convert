# -*- coding: utf-8 -*-

import cv2
import numpy as np


def get_image(img_path, resizeH=224, resizeW=224, resizeC=3, norm=True, meanB=104, meanG=117, meanR=123, std=1, rgb=False, nchw=False):
    import torchvision.transforms as transforms
    # image = np.asarray(image)
    # print(image.shape)
    # print(image[0, 0, 0:3])
    if norm:
        from PIL import Image
        input_size = 224
        image = Image.open(img_path).convert('RGB')
        if image is None:
            raise FileNotFoundError('No such image: {}'.format(img_path))
        transform = transforms.Compose(
            [transforms.Resize(int(input_size // 0.875)),
             transforms.CenterCrop(input_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ]
        )
        image = transform(image)
        image = image.detach().numpy()
        image = image.astype('float32')
        image = np.expand_dims(image, 0)
        image = np.transpose(image, axes=(0, 2, 3, 1)).copy()
    else:
        import cv2
        img = cv2.imread(img_path, flags=-1)
        if img is None:
            raise FileNotFoundError('No such image: {}'.format(img_path))
        try:
            img_dim = img.shape[2]
        except IndexError:
            img_dim = 1
        if img_dim == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img_dim == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img.astype('float32')
        new_size = 256
        resized_im = cv2.resize(img_float, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        image = resized_im[16:240, 16:240, :]
        image = np.round(image).astype('uint8')
        image = np.expand_dims(image, 0)
        # image = np.transpose(image, axes=(0, 2, 3, 1))
    return image


def image_preprocess(img_path, norm=True):
    return get_image(img_path, norm=norm)
