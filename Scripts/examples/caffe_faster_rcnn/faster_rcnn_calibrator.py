# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import numpy as np
import argparse
import json
import rpn
import pickle
from calibrator_custom import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='Faster RCNN Calibrator Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path.')
    parser.add_argument('-m0', '--model0', type=str, required=True,
                        help='Main Model path.')
    parser.add_argument('-m1', '--model1', type=str, required=True,
                        help='Second Model path.')
    parser.add_argument('--input_config0', type=str, required=True,
                        help='Main Model Input config path.')
    parser.add_argument('--input_config1', type=str, required=True,
                        help='Second Model Input config path.')
    parser.add_argument('--quant_level', type=str, default='L5',
                        choices=['L1', 'L2', 'L3', 'L4', 'L5'],
                        help='Indicate Quantilization level. The higher the level, the slower the speed and the higher the accuracy.')
    parser.add_argument('--num_process', default=10, type=int, help='Amount of processes run at same time.')
    parser.add_argument('-o0', '--output0', default=None, type=str, help='Output path for main fixed model.')
    parser.add_argument('-o1', '--output1', default=None, type=str, help='Output path for second fixed model.')

    return parser.parse_args()

def check_input_img(img_path):
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    h_w = height / width
    like_4_3 = lambda x: x + 3 - (x * 4 % 3)
    like_3_4 = lambda x: x + 4 - (x * 3 % 4)
    if h_w > 0.75:
        h = like_4_3(height)
        w = h * 4 // 3
        img_out = np.pad(img, ([0, h-height], [0, w-width], [0, 0]), 'constant')
        print('The picture: %s ratio is not 4: 3, the filled size is [H: %d, W: %d]' % (img_path, img_out.shape[0], img_out.shape[1]))
    elif h_w < 0.75:
        w = like_3_4(width)
        h = w * 3 // 4
        img_out = np.pad(img, ([0, h-height], [0, w-width], [0, 0]), 'constant')
        print('The picture: %s ratio is not 4: 3, the filled size is [H: %d, W: %d]' % (img_path, img_out.shape[0], img_out.shape[1]))
    else:
        img_out = img

    return img_out

def image_generator(image_list, preprocess_func):
    for image in image_list:
        img = preprocess_func(image)
        yield [img]

def fill_inputImg2main(img_out, resizeH=600, resizeW=800, resizeC=3, norm=True, meanB=102.9801, meanG=115.9465, meanR=122.7717, std=1, nchw=False):
    width = img_out.shape[1]
    height = img_out.shape[0]
    im_max = max(width, height)
    im_min = min(width, height)
    im_scale = resizeH / im_min
    if (im_scale * im_max) > resizeW:
        im_scale = resizeW / im_max

    img_norm = cv2.resize(img_out, (0, 0), fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    if norm:
        img_norm = (img_norm - [meanB, meanG, meanR]) / std
        img_norm = img_norm.astype('float32')

    if nchw:
        # NCHW
        img_norm = np.transpose(img_norm.reshape(resizeW, resizeH, -1), axes=(2, 0, 1))

    return np.expand_dims(img_norm, 0), im_scale

class Net(calibrator_custom.SIM_Calibrator):
    def __init__(self, main_model_path, main_input_config, second_model_path, second_input_config):
        super().__init__()
        self.main_model = calibrator_custom.calibrator(main_model_path, main_input_config)
        self.second_model = calibrator_custom.calibrator(second_model_path, second_input_config)
        self.rpn = rpn.ProposalLayer()

    def forward(self, x):
        out_details = self.main_model.get_output_details()
        input_data, im_scale = fill_inputImg2main(x)
        self.main_model.set_input(0, input_data)
        self.main_model.invoke()
        result_list = []
        for idx, _ in enumerate(out_details):
            result = self.main_model.get_output(idx)
            result_list.append(result)
        im_info = np.array([x.shape[0], x.shape[1], im_scale]).reshape(1, 3)
        bottom = [result_list[0], result_list[1], im_info]
        roi = self.rpn.forward(bottom)
        out2_details = self.second_model.get_output_details()
        self.second_model.set_input(0, result_list[2])
        self.second_model.set_input(1, roi)
        self.second_model.invoke()
        second_result = []
        for idx, _ in enumerate(out2_details):
            result = self.second_model.get_output(idx)
            second_result.append(result)
        return second_result

def main():
    args = arg_parse()
    image_path = args.image
    main_model_path = args.model0
    second_model_path = args.model1
    main_input_config = args.input_config0
    second_input_config = args.input_config1
    quant_level = args.quant_level
    num_subsets = args.num_process
    output0 = args.output0
    output1 = args.output1

    if not os.path.exists(main_model_path):
        raise FileNotFoundError('No such {} model'.format(main_model_path))
    if not os.path.exists(second_model_path):
        raise FileNotFoundError('No such {} model'.format(second_model_path))

    if not os.path.exists(main_input_config):
        raise FileNotFoundError('Main model input_config.ini file not found.')
    if not os.path.exists(second_input_config):
        raise FileNotFoundError('Second model input_config.ini file not found.')

    net = Net(main_model_path, main_input_config, second_model_path, second_input_config)
    print(net)

    if not os.path.exists(image_path):
        raise FileNotFoundError('No such {} image or directory.'.format(image_path))

    if os.path.isdir(image_path):
        image_list = [os.path.join(image_path, i) for i in os.listdir(image_path)]
        img_gen = image_generator(image_list, check_input_img)
    else:
        img_gen = image_generator([image_path], check_input_img)

    out_main_model = utils.get_out_model_name(main_model_path, output0)
    out_second_model = utils.get_out_model_name(second_model_path, output1)
    net.convert(img_gen, num_process=num_subsets, quant_level=quant_level, fix_model=[out_main_model, out_second_model])

if __name__ == '__main__':
    main()
