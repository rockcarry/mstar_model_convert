# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import numpy as np
import argparse
import pickle
from calibrator_custom import utils


class Net(calibrator_custom.SIM_Simulator):
    def __init__(self, model_path, phase, skip_convert_input_formats, log, input_config):
        super().__init__()
        self.skip_convert_input_formats = skip_convert_input_formats
        if phase == 'Float':
            self.model = calibrator_custom.float_simulator(model_path, show_log=log)
        elif phase == 'Fixed':
            self.model = calibrator_custom.fixed_simulator(model_path, show_log=log)
        else:
            self.model = calibrator_custom.offline_simulator(model_path, input_config=input_config, show_log=log)

    def forward(self, x):
        in_details = self.model.get_input_details()
        out_details = self.model.get_output_details()
        for idx, _ in enumerate(in_details):
            if self.skip_convert_input_formats:
                self.model.set_input(idx, x[idx])
            else:
                self.model.set_input(idx, utils.convert_to_input_formats(x[idx], in_details[idx]))
        self.model.invoke()
        result_list = []
        for idx, _ in enumerate(out_details):
            result = self.model.get_output(idx)
            # for Fixed and Offline model
            if result.shape[-1] != out_details[idx]['shape'][-1]:
                result = result[..., :out_details[idx]['shape'][-1]]
            if out_details[idx]['dtype'] == np.int16:
                scale, _ = out_details[idx]['quantization']
                result = np.dot(result, scale)
            result_list.append(result)
        return result_list


def arg_parse():
    parser = argparse.ArgumentParser(description='Simulator Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('-t', '--type', type=str, required=calibrator_custom.__version__[:2] in ['1.', 'Q_'],
                        choices=['Float', 'Fixed', 'Offline'],
                        help='Indicate model data type.')
    parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    parser.add_argument('--input_config', type=str, default=None,
                        help='Input config path.')
    parser.add_argument('--skip_convert_input_formats', default=False, action='store_true',
                        help='Switch whether to skip convert training_input_formats into input_formats for model input, default=(False)')
    parser.add_argument('--num_process', default=1, type=int, help='Amount of processes run at same time.')
    parser.add_argument('--show_log', default=False, action='store_true', help='Show log on screen.')

    return parser.parse_args()


def main():
    args = arg_parse()
    image_path = args.image
    model_path = args.model
    phase = args.type
    model_name = args.preprocess
    skip_convert_input_formats = args.skip_convert_input_formats
    input_config = args.input_config
    num_subsets = args.num_process
    log = args.show_log

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))
    if input_config is not None and not os.path.exists(input_config):
        raise FileNotFoundError('input_config.ini file not found.')

    net = Net(model_path, phase, skip_convert_input_formats, log, input_config)
    norm = True if net.model.class_type in ['FLOAT', 'CMODEL_FLOAT'] else False
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
        img_gen = utils.image_generator(image_list, preprocess_funcs, norm)
    elif os.path.basename(base_name).split('.')[-1].lower() in utils.image_suffix:
        img_gen = utils.image_generator([base_name], preprocess_funcs, norm)
    else:
        with open(base_name, 'r') as f:
            multi_images = f.readlines()
        if dir_name is None:
            multi_images = [images.strip().split(',') for images in multi_images]
        else:
            multi_images = [[os.path.join(dir_name, i) for i in images.strip().split(',')] for images in multi_images]
        img_gen = utils.image_generator(multi_images, preprocess_funcs, norm)

    result = net(img_gen, num_process=num_subsets)

    # add some postprocess mothods for `result`
    result_file = os.path.basename(model_path) + '_' + os.path.basename(image_path) + '.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()
