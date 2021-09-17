# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import copy
import numpy as np
import argparse
import itertools
import pickle
from calibrator_custom import utils


class Float_Model(calibrator_custom.SIM_Simulator):
    def __init__(self, model_path):
        super().__init__()
        self.model = calibrator_custom.float_simulator(model_path)

    def forward(self, x):
        in_details = self.model.get_input_details()
        out_details = self.model.get_output_details()
        for idx, _ in enumerate(in_details):
            self.model.set_input(idx, x[idx])
        self.model.invoke()
        result_list = []
        for idx, _ in enumerate(out_details):
            result = self.model.get_output(idx)
            result_list.append(result)
        return result_list


class Net(calibrator_custom.SIM_Calibrator):
    def __init__(self, model_path, input_config):
        super().__init__()
        self.model = calibrator_custom.calibrator(model_path, input_config)

    def forward(self, x):
        in_details = self.model.get_input_details()
        out_details = self.model.get_output_details()
        for idx, _ in enumerate(in_details):
            self.model.set_input(idx, x[idx])
        self.model.invoke()
        result_list = []
        for idx, _ in enumerate(out_details):
            result = self.model.get_output(idx)
            result_list.append(result)
        return result_list


def arg_parse():
    parser = argparse.ArgumentParser(description='Optimizer Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    parser.add_argument('--num_process', default=10, type=int, help='Amount of processes run at same time.')

    return parser.parse_args()


def main():
    args = arg_parse()
    image_path = args.image
    model_path = args.model
    input_config = args.input_config
    model_name = args.preprocess
    num_subsets = args.num_process

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))
    if not os.path.exists(input_config):
        raise FileNotFoundError('input_config.ini file not found.')

    net = Net(model_path, input_config)
    print(net)

    if not os.path.exists(image_path):
        raise FileNotFoundError('No such {} image or directory.'.format(image_path))

    preprocess_funcs = [utils.image_preprocess_func(n) for n in model_name.split(',')]
    if os.path.isdir(image_path):
        image_list = utils.all_path(image_path)
        img_gen = utils.image_generator(image_list, preprocess_funcs)
    elif os.path.basename(image_path).split('.')[-1].lower() in utils.image_suffix:
        img_gen = utils.image_generator([image_path], preprocess_funcs)
    else:
        with open(image_path, 'r') as f:
            multi_images = f.readlines()
        multi_images = [images.strip().split(',') for images in multi_images]
        img_gen = utils.image_generator(multi_images, preprocess_funcs)

    gen_optim, gen_float, gen_imgs = itertools.tee(img_gen, 3)
    optim_net = Net(model_path, input_config)
    optim_info = optim_net.convert_optim_model(gen_optim, num_process=num_subsets)

    float_sim = Float_Model(model_path)
    float_golden = float_sim(gen_float, num_process=num_subsets)

    compression_rates = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    quant_infos = net.optimize(gen_imgs, optim_info, float_golden, compression_rates, num_subsets)
    for idx, rate in enumerate(compression_rates):
        with open('optimized_quant_info_{}_{}.pkl'.format(idx, rate), 'wb') as f:
            pickle.dump(quant_infos[idx], f)


if __name__ == '__main__':
    main()
