# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import numpy as np
import argparse
from calibrator_custom import utils
from calibrator_custom import rpc_simulator
from utils import misc


class Net(calibrator_custom.RPC_Simulator):
    def __init__(self, model_path, core_id):
        super().__init__()
        self.model = rpc_simulator.simulator(model_path)
        self.core_id = core_id

    def forward(self, x):
        in_details = self.model.get_input_details()
        out_details = self.model.get_output_details()
        for idx, _ in enumerate(in_details):
            self.model.set_input(idx, utils.convert_to_input_formats(x[idx], in_details[idx]))
        self.model.invoke(core_id=self.core_id)
        result_list = []
        for idx, _ in enumerate(out_details):
            result = self.model.get_output(idx)
            # for Fixed and Offline model
            if result.shape[-1] != out_details[idx]['shape'][-1]:
                result = result[..., :out_details[idx]['shape'][-1]]
            if out_details[idx]['dtype'] == np.int16:
                scale, _ = out_details[idx]['quantization']
                result = np.multiply(result, scale)
            result_list.append(result)
        return result_list


def arg_parse():
    parser = argparse.ArgumentParser(description='RPC Simulator Tool')
    parser.add_argument('--host', type=str, required=True,
                        help='IPU Server host.')
    parser.add_argument('--port', type=int, required=True,
                        help='IPU Server port.')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Offline Model path.')
    parser.add_argument('-c', '--category', type=str, default='Unknown',
                        choices=['Classification', 'Detection', 'Unknown'],
                        help='Indicate net category.')
    parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    parser.add_argument('--core_id', type=int, default=0,
                        help='Core id for IPU run.')
    parser.add_argument('--draw_result', default=None, type=str, help='Output directory of draw bbox images.')
    parser.add_argument('--continue_run', default=False, action='store_true', help='Continue run datasets.')

    return parser.parse_args()


def main():
    args = arg_parse()
    host = args.host
    port = args.port
    image_path = args.image
    model_path = args.model
    category = args.category
    model_name = args.preprocess
    core_id = args.core_id
    draw_result = args.draw_result
    continue_run = args.continue_run

    if 'SGS_IPU_DIR' in os.environ:
        move_log = misc.Move_Log(clean=True)
    elif 'TOP_DIR' in os.environ:
        move_log = misc.Move_Log(clean=False)
    else:
        raise OSError('\033[31mRun source cfg_env.sh in top directory.\033[0m')

    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        dir_name = None
        base_name = image_path
    if not os.path.exists(base_name):
        raise FileNotFoundError('No such {} image or directory.'.format(base_name))

    is_read_img_list = False
    preprocess_funcs = [utils.image_preprocess_func(n) for n in model_name.split(',')]
    if os.path.isdir(base_name):
        image_list = utils.all_path(base_name)
    elif os.path.basename(base_name).split('.')[-1].lower() in utils.image_suffix:
        image_list = [os.path.abspath(base_name)]
    else:
        is_read_img_list = True
        with open(base_name, 'r') as f:
            image_list = f.readlines()
        if dir_name is None:
            image_list = [[os.path.abspath(im) for im in images.strip().split(',')] for images in image_list]
        else:
            image_list = [[os.path.abspath(os.path.join(dir_name, i)) for i in images.strip().split(',')] for images in image_list]

    if continue_run:
        if is_read_img_list:
            raise ValueError('\033[31mNot support image_list for `--continue_run` input! Can not continue run!\033[0m')
        if os.path.exists('output'):
            results_list = os.listdir('output')
        elif os.path.exists('log/output'):
            shutil.move('log/output', os.getcwd())
            results_list = os.listdir('output')
        else:
            raise FileNotFoundError('\033[31mNo `output` directory, do not use `--continue_run` param.\033[0m')
        prefix_result = '{}_{}_'.format(category, os.path.basename(model_path))
        image_dict = dict([(os.path.basename(im), im) for im in image_list])
        if len(image_dict) != len(image_list):
            raise ValueError('\033[31mFind input file with same name! Can not continue run!\033[0m')
        results_list = [i[len(prefix_result): -4] for i in results_list if i.split('.')[-1] == 'txt']
        results_list = [image_dict[i] for i in results_list]
        results_set = set(results_list)
        img_set = set(image_list)
        img_run = img_set - results_set
        image_list = list(img_run)
        print('\033[92m{} images already ran, left {} images to run continue.\033[0m'.format(len(results_list), len(image_list)))
    else:
        misc.renew_folder('output')

    rpc_simulator.connect(host, port)
    net = Net(model_path, core_id)
    print(net)
    out_details = net.model.get_output_details()
    show_result = False if len(image_list) > 1 else True
    img_gen = utils.image_generator(image_list, preprocess_funcs, False)
    result = net(img_gen)

    for idx, img_path in enumerate(image_list):
        result_list = result[idx]
        if category == 'Detection':
            misc.postDetection(model_path, img_path, result_list, out_details, draw_result, show_result)
        elif category == 'Classification':
            misc.postClassification(model_path, img_path, result_list, out_details, show_result)
        else:
            misc.postUnknown(model_path, img_path, result_list, out_details, True)


if __name__ == '__main__':
    main()
