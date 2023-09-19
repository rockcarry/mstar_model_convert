# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import numpy as np
import argparse
import json
import pickle
from calibrator_custom import utils


class Net(calibrator_custom.SIM_Calibrator):
    def __init__(self, model_path, input_config, core_mode, log):
        super().__init__()
        self.model = calibrator_custom.calibrator(model_path, input_config, work_mode=core_mode, show_log=log)

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
    parser.add_argument('--quant_level', type=str, default='L5',
                        choices=['L1', 'L2', 'L3', 'L4', 'L5'],
                        help='Indicate Quantilization level. The higher the level, the slower the speed and the higher the accuracy.')
    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        parser.add_argument('--work_mode', type=str, default=None,
                            choices=['single_core', 'multi_core'],
                            help='Indicate calibrator work_mode.')
    parser.add_argument('--mixed_precision', type=str, default='sp',
                        help='Indicate strategy of mixed precision.')
    parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    parser.add_argument('--num_process', default=10, type=int, help='Amount of processes run at same time.')
    parser.add_argument('--show_log', default=False, action='store_true', help='Show log on screen.')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output path for fixed model.')

    return parser.parse_args()


def main():
    args = arg_parse()
    image_path = args.image
    model_path = args.model
    input_config = args.input_config
    quant_level = args.quant_level
    mixed_precision = args.mixed_precision
    quant_file = args.quant_file
    model_name = args.preprocess
    num_subsets = args.num_process
    log = args.show_log
    output = args.output
    work_mode = None
    if calibrator_custom.utils.VERSION[:2] in ['S6'] and args.work_mode is not None:
        work_mode = args.work_mode

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))

    if not os.path.exists(input_config):
        raise FileNotFoundError('input_config.ini file not found.')

    net = Net(model_path, input_config, work_mode, log)
    print(net)
    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        dir_name = None
        base_name = image_path
    if not os.path.exists(base_name):
        raise FileNotFoundError('No such {} image or directory.'.format(base_name))

    preprocess_funcs = [utils.image_preprocess_func(n) for n in model_name.split(',')]
    if os.path.isdir(base_name):
        image_list = utils.all_path(base_name)
        img_gen = utils.image_generator(image_list, preprocess_funcs)
    elif os.path.basename(base_name).split('.')[-1].lower() in utils.image_suffix:
        img_gen = utils.image_generator([base_name], preprocess_funcs)
    else:
        with open(base_name, 'r') as f:
            multi_images = f.readlines()
        if dir_name is None:
            multi_images = [images.strip().split(',') for images in multi_images]
        else:
            multi_images = [[os.path.join(dir_name, i) for i in images.strip().split(',')] for images in multi_images]
        img_gen = utils.image_generator(multi_images, preprocess_funcs)

    quant_param = None
    if quant_file is not None:
        if not os.path.exists(quant_file):
            raise FileNotFoundError('No such quant_file: {}'.format(quant_file))
        else:
            try:
                with open(quant_file, 'rb') as f:
                    quant_param = pickle.load(f)
            except pickle.UnpicklingError:
                with open(quant_file, 'r') as f:
                    quant_param = json.load(f)
            except json.JSONDecodeError:
                raise ValueError('quant_param only support JSON or Pickle file.')
    out_model = utils.get_out_model_name(model_path, output)
    net.convert(img_gen, num_process=num_subsets, quant_level=quant_level, mixed_precision=mixed_precision,
                quant_param=quant_param, fix_model=[out_model])


if __name__ == '__main__':
    main()
