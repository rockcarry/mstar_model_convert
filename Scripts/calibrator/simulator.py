# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import shutil
import numpy as np
import argparse
import pickle
from calibrator_custom import utils
from utils import accuracy
from utils import misc


def arg_parse():
    parser = argparse.ArgumentParser(description='Simulator Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('-c', '--category', type=str, default='Unknown',
                        choices=['Classification', 'Detection', 'Unknown'],
                        help='Indicate net category.')
    if 'TOP_DIR' in os.environ:
        parser.add_argument('-t', '--type', type=str, required=calibrator_custom.__version__[:2] in ['1.', 'Q_'],
                            choices=['Float', 'Cmodel_float', 'Fixed', 'Fixed_without_ipu_ctrl', 'Offline'],
                            help='Indicate model data type.')
    else:
        parser.add_argument('-t', '--type', type=str, required=calibrator_custom.__version__[:2] in ['1.', 'Q_'],
                            choices=['Float', 'Fixed', 'Offline'],
                            help='Indicate model data type.')
    parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    parser.add_argument('-l', '--label', type=str, default=None,
                        help='Label for single image test / validation label path for datasets.')
    parser.add_argument('--input_config', type=str, default=None,
                        help='Input config path.')
    parser.add_argument('--tool', default=None, type=str, help='sgs_simulator path.')
    parser.add_argument('--num_process', default=1, type=int, help='Amount of processes run at same time.')
    parser.add_argument('--show_log', default=False, action='store_true', help='Show log on screen.')
    parser.add_argument('--dump_rawdata', default=False, action='store_true',
                        help='Switch whether to dump input raw data, default=(False)')
    parser.add_argument('--skip_garbage', default=False, action='store_true',
                        help='Switch whether to skip output garbage data, default=(False)')
    parser.add_argument('--draw_result', default=None, type=str, help='Output directory of draw bbox images.')
    parser.add_argument('--continue_run', default=False, action='store_true', help='Continue run datasets.')

    return parser.parse_args()


def main():
    args = arg_parse()
    image_path = args.image
    model_path = args.model
    category = args.category
    phase = args.type
    model_name = args.preprocess
    input_config = args.input_config
    num_subsets = args.num_process if args.num_process < os.cpu_count() else os.cpu_count()
    dump_rawdata = args.dump_rawdata
    skip_garbage = args.skip_garbage
    draw_result = args.draw_result
    continue_run = args.continue_run
    tool = args.tool
    misc.SHOW_LOG = args.show_log

    if 'SGS_IPU_DIR' in os.environ:
        move_log = misc.Move_Log(clean=True)
    elif 'TOP_DIR' in os.environ:
        move_log = misc.Move_Log(clean=False)
    else:
        raise OSError('\033[31mRun source cfg_env.sh in top directory.\033[0m')

    if not os.path.exists(model_path):
        raise FileNotFoundError('\033[31mNo such {} model\033[0m'.format(model_path))
    if input_config is not None and not os.path.exists(input_config):
        raise FileNotFoundError('\033[31minput_config.ini file not found.\033[0m')

    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        dir_name = None
        base_name = image_path
    if not os.path.exists(base_name):
        raise FileNotFoundError('\033[31mNo such {} image or directory.\033[0m'.format(base_name))

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

    if phase in ['Fixed', 'Offline']:
        try:
            if calibrator_custom.__version__[:2] in ['1.', 'Q_']:
                from calibrator_custom._fixedsim import fixed_simulator
                from calibrator_custom._offlinesim import offline_simulator
        except ModuleNotFoundError:
            print('\033[33m[WARNING] `python3` executes simulator.py for Fixed and Offline models is deprecated. ' \
                  'Please use `python32` to execute the simulator.py for Fixed and Offline models.\033[0m')

            if 'SGS_IPU_DIR' in os.environ:
                Project_path = os.environ['SGS_IPU_DIR']
                simulator = accuracy.which_simulator(Project_path, phase) if tool is None else tool
            else:
                Project_path = os.environ['TOP_DIR']
                simulator = misc.find_path(Project_path, 'label_image') if tool is None else tool

            eliminate = ''
            if skip_garbage:
                eliminate = eliminate + '--skip_garbage '
            if dump_rawdata:
                eliminate = eliminate + '--dump_rawdata '
            label_file = misc.Fake_Label(os.path.basename(model_path).split('.')[0])
            label = label_file.label_name
            print('\033[31mStart to evaluate on {}...\033[0m'.format(base_name.strip().split('/')[-2]
                if base_name.strip().split('/')[-1] == '' else base_name.strip().split('/')[-1]))
            if num_subsets < 2:
                accuracy.label_image(image_list, os.path.abspath(model_path), label, category, model_name, \
                    preprocess_funcs, Project_path, simulator, phase, draw_result, eliminate, continue_run)
            else:
                accuracy.run_simulator(image_list, label, os.path.abspath(model_path), category, model_name, \
                                       preprocess_funcs, Project_path, simulator, phase, eliminate, \
                                       continue_run, num_subsets=num_subsets)
            print('\033[31mRun evaluation OK.\033[0m')
            return

    misc.run_simulator(base_name, image_list, phase, model_path, category, preprocess_funcs, num_subsets,
                       dump_rawdata, draw_result, skip_garbage, input_config)


if __name__ == '__main__':
    main()
