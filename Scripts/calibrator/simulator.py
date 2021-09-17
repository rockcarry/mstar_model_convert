# -*- coding: utf-8 -*-

import os
import argparse
import shutil
from utils import accuracy
from utils import misc


def arg_parse():
    parser = argparse.ArgumentParser(description='Simulator Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('-l', '--label', type=str, default='0',
                        help='Label for single image test / validation label path for datasets.')
    parser.add_argument('-c', '--category', type=str, required=True,
                        choices=['Classification', 'Detection', 'Unknown'],
                        help='Indicate net category.')
    parser.add_argument('-t', '--type', type=str, required=True,
                        choices=['Float', 'Fixed', 'Offline'],
                        help='Indicate model data type.')
    parser.add_argument('-n', '--preprocess', type=str, default='0',
                        help='Name of model to select image preprocess method')
    parser.add_argument('--tool', default='0', type=str, help='sgs_simulator path.')
    parser.add_argument('--num_process', default=0, type=int, help='Amount of processes run at same time.')
    parser.add_argument('--dump_rawdata', default=False, action='store_true',
                        help='Switch whether to dump input raw data, default=(False)')
    parser.add_argument('--skip_garbage', default=False, action='store_true',
                        help='Switch whether to skip output garbage data, default=(False)')
    parser.add_argument('--draw_result', default='0', type=str, help='Output directory of draw bbox images.')
    parser.add_argument('--continue_run', default=False, action='store_true', help='Continue run datasets.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
    parser.add_argument('--save_input', default=False, action='store_true',
                        help='Switch whether to save input files, default=(False)')

    return parser.parse_args()


def main():
    args = arg_parse()
    image_path = args.image
    model = args.model
    label = args.label
    category = args.category
    phase = args.type
    model_name = args.preprocess
    tool = args.tool
    num_subsets = args.num_process
    skip_garbage = args.skip_garbage
    dump_rawdata = args.dump_rawdata
    draw_result = args.draw_result
    save_in = args.save_input
    continue_run = args.continue_run

    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        dir_name = None
        base_name = image_path
    if not os.path.exists(base_name):
        raise FileNotFoundError('No such {} image or directory.'.format(base_name))

    img_list = []
    if os.path.isdir(base_name):
        img_list = sorted(misc.all_path(os.path.abspath(base_name)))
    elif os.path.basename(base_name).split('.')[-1].lower() in misc.image_suffix:
        img_list.append(os.path.abspath(base_name))
    else:
        with open(base_name, 'r') as f:
            img_list = f.readlines()
        if dir_name is None:
            img_list = [[os.path.abspath(img) for img in images.strip().split(',')] for images in img_list]
        else:
            img_list = [[os.path.join(dir_name, img) for img in images.strip().split(',')] for images in img_list]

    dataset_size = len(img_list)
    if continue_run:
        if os.path.exists('output'):
            results_list = os.listdir('output')
        elif os.path.exists('log/output'):
            shutil.move('log/output', os.getcwd())
            results_list = os.listdir('output')
        else:
            raise FileNotFoundError('\033[31mNo `output` directory, do not use `--continue_run` param.\033[0m')
        prefix_result = '{}_{}_'.format(category, os.path.basename(model))
        results_list = [i[len(prefix_result): -4] for i in results_list if i.split('.')[-1] == 'txt']
        results_list = [misc.find_path(base_name, i) for i in results_list]
        results_set = set(results_list)
        img_set = set(img_list)
        img_run = img_set - results_set
        img_list = list(img_run)
        print('\033[31m{} images already ran, left {} images to run continue.\033[0m'.format(len(results_list), len(img_list)))

    try:
        debug = args.debug
    except:
        debug = False

    if debug and num_subsets > 1:
            raise OSError('\033[31mCan not run debug mode in mutiple process, only `--num_process 1` can run debug mode.\033[0m')

    if not os.path.exists(model):
        raise FileNotFoundError('No such model: {}'.format(model))
    model = os.path.abspath(model)

    max_cpu_count = os.cpu_count()
    if (num_subsets > len(img_list)) and (not continue_run):
        num_subsets = len(img_list)
    if num_subsets > max_cpu_count:
        num_subsets = max_cpu_count
        print('\033[33m[WARNING] Only {} CPU cores can run currently!\033[0m'.format(max_cpu_count))

    if label != '0':
        if not os.path.exists(label):
            raise FileNotFoundError('No such label: {}'.format(label))
        label = os.path.abspath(label)
    else:
        if (num_subsets > 1) and ('SGS_IPU_DIR' in os.environ):
            raise OSError('\033[31mMulti-process mode must need datasets label.\033[0m')
        label_file = misc.Fake_Label(os.path.basename(model).split('.')[0])
        label = label_file.label_name

    preprocess_func = None
    if model_name != '0':
        preprocess_func = [misc.image_preprocess_func(n) for n in model_name.split(',')]

    eliminate = ''
    if skip_garbage:
        eliminate = eliminate + '--skip_garbage '
    if dump_rawdata:
        eliminate = eliminate + '--dump_rawdata '

    if 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        move_log = misc.Move_Log(clean=True)
        if tool == '0':
            simulator = accuracy.which_simulator(Project_path, phase)
        else:
            simulator = tool
    elif 'TOP_DIR' in os.environ:
        Project_path = os.environ['TOP_DIR']
        move_log = misc.Move_Log(clean=False)
        if tool == '0':
            simulator = misc.find_path(Project_path, 'label_image')
        else:
            simulator = tool
    else:
        raise OSError('Run source cfg_env.sh in top directory.')

    print('\033[31mStart to evaluate on {}...\033[0m'.format(base_name.strip().split('/')[-2]
          if base_name.strip().split('/')[-1] == '' else base_name.strip().split('/')[-1]))
    if num_subsets < 2:
        accuracy.label_image(img_list, model, label, category, model_name, preprocess_func, Project_path, simulator, phase, draw_result, eliminate, continue_run, debug, save_in)
    else:
        accuracy.run_simulator(img_list, label, model, category, model_name, preprocess_func, Project_path, simulator,
                               phase, eliminate, continue_run, dataset_size, save_in, num_subsets)
    print('\033[31mRun evaluation OK.\033[0m')


if __name__ == '__main__':
    main()
