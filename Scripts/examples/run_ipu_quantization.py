# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import argparse
import json
import calibrator_custom
from calibrator_custom import utils
import torch
import pickle
from calibrator_custom.ipu_quantization_lib import *


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
    parser = argparse.ArgumentParser(description='Calibrator Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    parser.add_argument('-c', '--category', type=str, default='Unknown',
                        choices=['Classification', 'Detection', 'Unknown'],
                        help='Indicate net category.')
    parser.add_argument('--quant_level', type=str, default='L5',
                        choices=['L1', 'L2', 'L3', 'L4', 'L5'],
                        help='Indicate Quantilization level. The higher the level, the slower the speed and the higher the accuracy.')
    parser.add_argument('--quant_precision', type=int, default=8,
                        help='Set unified quant precision for CONV2D input and weights.')
    parser.add_argument('--mp_level', type=int, default=0,
                        help='Indicate strategy of mixed precision. 1 for more conservative choice, 2 for more compression size policy.')
    parser.add_argument('--mp_rate', type=float, default=1.0,
                        help='Set maximum mixed precision CONV2D weights compression rate. Should be between 0.5 and 1.0.')
    parser.add_argument('--quant_file', type=str, default=None,
                        help='Save path for quant_params')
    parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    parser.add_argument('--num_process', default=10, type=int, help='Amount of processes run at same time.')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output path for fixed model.')

    return parser.parse_args()


def main():
    args = arg_parse()
    image_path = args.image
    model_path = args.model
    input_config = args.input_config
    quant_level = args.quant_level
    mp_level = args.mp_level
    mp_rate = args.mp_rate
    quant_file = args.quant_file
    preprocess = args.preprocess
    num_subsets = args.num_process
    output = args.output

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))

    if not os.path.exists(input_config):
        raise FileNotFoundError('input_config.ini file not found.')

    net = Net(model_path, input_config)
    print(net)

    if not os.path.exists(image_path):
        raise FileNotFoundError('No such {} image or directory.'.format(image_path))
    op_details = net.model.get_op_details()

    onnx_file = os.path.join(os.path.dirname(model_path), os.path.basename(model_path).replace('_float.sim', '.onnx'))
    quant_cfg = get_quant_config(
        onnx_file,
        input_config=input_config, python_file=preprocess, calset_dir=image_path,
        quant_rules=op_details
    )

    print('\033[92mRun Quantization using {}\033[0m'.format(str(quant_cfg.device)))
    if mp_level > 0 or mp_rate < 1.0:
        if mp_level > 0:
            recommend = mp_level
            assert recommend in [1, 2]
            sim_params = mp_quantize(quant_cfg, recommend=recommend, model_type=args.category, retrain=0)
        else:
            assert 0 < mp_rate < 1.0
            sim_params = mp_quantize(quant_cfg, mp_rate=mp_rate, model_type=args.category, retrain=0)
    else:
        quant_loss, quant_params = quantize(
            quant_cfg,
            quant_precision=args.quant_precision, retrain=0, verbose=False
        )
        sim_params = quant_params['sim']
    if quant_file is not None:
        with open(quant_file, 'wb') as fw:
            pickle.dump(sim_params, fw)

    preprocess_funcs = [utils.image_preprocess_func(n) for n in preprocess.split(',')]
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

    out_model = utils.get_out_model_name(model_path, output)
    print('\033[92mFeed sim_params\033[0m')
    sim_params = pickle.load(open(quant_file, 'rb'))
    net.convert(img_gen, num_process=num_subsets, quant_level=quant_level,
                quant_param=sim_params, fix_model=[out_model])
    print('\033[92mFixed model generated!\033[0m')


if __name__ == '__main__':
    main()
