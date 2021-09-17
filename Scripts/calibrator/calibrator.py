# -*- coding: utf-8 -*-

import cv2
import os
import argparse
import shutil
from utils import statistics
from utils import misc
from utils.quantization.quantize import quantize
import time
import pdb
import re
import sys
from multiprocessing import Process


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
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        parser.add_argument('-p', '--phase', type=str, required=True,
                            choices=['Statistics','Fixed', 'Fixed_without_IPU'],
                            help='Indicate calibration phase.')
        parser.add_argument('--statistics_type', type=str, default='QIO,QKI',
                            help='Indicate Statistics Type. one of QIK/QIO,QKP/QKI,None, or concat like as QIK,QKI')
        parser.add_argument('--quant_threshold', default=1024, type=int, help='Convolution quant scale ratio threshold.')
        parser.add_argument('--quant_params_file', default=None, type=str, help='Precomputed quantization params in .pkl format')

        parser.add_argument('--ib_strategy', type=str, default=None,
                            help='ib strategy.')
        parser.add_argument('--kb_strategy', type=str, default=None,
                            help='kb strategy.')
    elif ('TOP_DIR' in os.environ) or ('SGS_IPU_DIR' in os.environ):
        parser.add_argument('--quant_level', type=str, default='L5',
                            choices=['L1', 'L2', 'L3', 'L4', 'L5'],
                            help='Indicate Quantilization level. The higher the level, the slower the speed and the higher the accuracy.')
    # ------ added by tao.xu 2020.07.24
    parser.add_argument('-sth', '--sensitivity_threshold', default=0.002, type=float, help='Sensitivity threshold (valid if 0<st<1) for upgrading to 16 bit quantization.')
    # ------- added by tao.xu 2020.07.24

    parser.add_argument('-n', '--preprocess', type=str, default='0',
                        help='Name of model to select image preprocess method')
    parser.add_argument('--num_process', default=10, type=int, help='Amount of processes run at same time.')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output path for fixed model.')
    parser.add_argument('-t', '--tool', default='0', type=str, help='sgs_calibration path.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
    parser.add_argument('--save_input', default=False, action='store_true',
                        help='Switch whether to save input files, default=(False)')

    return parser.parse_args()


def main():
    args = arg_parse()
    image_path = args.image
    model = args.model
    input_config = args.input_config
    category = 'Unknown'
    model_name = args.preprocess
    tool_path = args.tool
    num_subsets = args.num_process
    save_in = args.save_input
    output = args.output
    sth = args.sensitivity_threshold

    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        dir_name = None
        base_name = image_path
    if not os.path.exists(base_name):
        raise FileNotFoundError('No such {} image or directory.'.format(base_name))
    if not os.path.exists(model):
        raise FileNotFoundError('No such model: {}'.format(model))

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

    model = os.path.abspath(model)

    preprocess_func = None
    if model_name != '0':
        preprocess_func = [misc.image_preprocess_func(n) for n in model_name.split(',')]

    max_cpu_count = os.cpu_count()
    if num_subsets > len(img_list):
        num_subsets = len(img_list)
    if num_subsets > max_cpu_count:
        num_subsets = max_cpu_count
        print('\033[33m[WARNING] Only {} CPU cores can run currently!\033[0m'.format(max_cpu_count))

    if 'SGS_IPU_DIR' in os.environ:
        quant_level = args.quant_level
        quant_threshold = 1024
        Project_path = os.environ['SGS_IPU_DIR']
        Dev_path = os.path.join(Project_path, 'cfg')
        debug_info = False
        label = misc.Fake_Label(os.path.basename(model).split('.')[0])
        move_log = misc.Move_Log()
        if tool_path == '0':
            tool_path = misc.find_path(Project_path, 'sgs_calibrator')
        if os.path.exists('tensor_min_max.txt'):
            os.remove('tensor_min_max.txt')
        if os.path.exists('tensor_statistics.txt'):
            os.remove('tensor_statistics.txt')
        if os.path.exists('tensor_statistics_type.txt'):
            os.remove('tensor_statistics_type.txt')
        if os.path.exists('tensor_weight.txt'):
            os.remove('tensor_weight.txt')
        if os.path.exists('tensor_weight_calibration.txt'):
            os.remove('tensor_weight_calibration.txt')
        if os.path.exists('tensor_qab.txt'):
            os.remove('tensor_qab.txt')
        if os.path.exists('tensor_chn.txt'):
            os.remove('tensor_chn.txt')
        if os.path.exists('softmax_lut_range.txt'):
            os.remove('softmax_lut_range.txt')
        use_qio = False
        use_qkp = False
        use_qki = False
        if quant_level == 'L2':
            use_qkp = True
        elif quant_level == 'L3':
            use_qio = True
            use_qkp = True
        elif quant_level == 'L4':
            use_qkp = True
            use_qki = True
        elif quant_level == 'L5':
            use_qio = True
            use_qkp = True
            use_qki = True

        if use_qkp or use_qio:
            statistics.sgs_calibration_statistics('qkp', input_config, img_list, label.label_name, model, category, tool_path, model_name, preprocess_func,
                                                    debug_info, save_in)

        print('\033[31mStart to analysis model...\033[0m')
        if num_subsets < 2:
            statistics.sgs_calibration_statistics('minmax', None, img_list, label.label_name, model, category, tool_path, model_name, preprocess_func, save_in=save_in)
        else:
            statistics.run_statistics_minmax(img_list, label.label_name, model, category, model_name, preprocess_func, Project_path, tool_path,
                                             save_in, num_subsets=num_subsets)
        print('\033[31mRun analysis model OK.\033[0m')
        cmodel_float_model = model
        if use_qio or use_qkp:
            print('\033[31mStart to run statistics...\033[0m')
            if num_subsets < 2:
                statistics.sgs_calibration_statistics('histogram', None, img_list, label.label_name, model, category, tool_path, model_name, preprocess_func, debug_info, save_in)
            else:
                statistics.run_statistics_kl(img_list, label.label_name, model, category, model_name, preprocess_func, Project_path, tool_path,
                                                save_in, num_subsets=num_subsets)
            print('\033[31mRun statistics OK.\033[0m')
            if use_qio:
                quantize_method = 'qio'
            else:
                quantize_method = None
            quantize(tensors_file ='./tensor_weight.txt', by_channel=use_qkp, tensor_statistics_file='./tensor_statistics.txt', quantize_method=quantize_method,
                        minmax_file='./tensor_min_max.txt', parallel=num_subsets, verbose=False, suggest_type=use_qki, sth=sth)

        m_p = 'tensor_min_max.txt'
        o_p = 'tensor_chn.txt'
        s_p = 'softmax_lut_range.txt'
        cmodel_float_model = statistics.sgs_calibration_fixed('Cmodel_float', model, tool_path, Dev_path, input_config, m_p, o_p, s_p, output)

        if os.path.exists('statistics'):
            shutil.rmtree('statistics')
        print('\033[31mRun analysis images OK.\033[0m')
        print('\033[31mStart to convert model...\033[0m')
        statistics.sgs_calibration_fixed('Fixed', cmodel_float_model, tool_path, Dev_path, input_config, None, None, None, output, quantThreshold=quant_threshold)
        print('\033[31mRun convert model OK.\033[0m')

    elif 'TOP_DIR' in os.environ:
        phase = args.phase
        debug = args.debug
        quant_threshold = args.quant_threshold
        ib_strategy = args.ib_strategy
        kb_strategy = args.kb_strategy
        statistics_type = args.statistics_type
        use_qio = False
        use_qkp = False
        use_qki = False
        if phase == 'Statistics' and statistics_type != '0':
            for Type in statistics_type.split(','):
                if Type == 'QIO':
                    use_qio = True
                elif Type == 'QKP':
                    use_qkp = True
                elif Type == 'QKI':
                    use_qkp = True
                    use_qki = True
                elif Type == 'None':
                    use_qio = False
                    use_qkp = False
                    use_qki = False
                    break
                else:
                    raise NameError('invalid statistics_type choice: \'{}\'.'.format(Type))

        if debug and num_subsets > 1:
            raise OSError('\033[31mCan not run debug mode in mutiple process, only `--num_process 1` can run debug mode.\033[0m')

        elif debug and phase == 'Statistics':
            debug_list = ['minmax', 'convert_cmodel_float']
            if use_qio:
                debug_list.append('histogram')
            if use_qkp:
                debug_list.append('qkp')
            str_list = ''
            for item in debug_list:
                str_list += item
                if item != debug_list[len(debug_list)-1]:
                    str_list += ', '
            debug_info = input('Which Statistics phase do you wanna debug? [{}] '.format(str_list))
            if debug_info not in debug_list:
                raise OSError('Only {} phase support debug.'.format(str_list))
        else:
            debug_info = '0'

        Project_path = os.environ['TOP_DIR']
        Dev_path = os.path.join(Project_path, '../SRC/Tool/cfg')
        if tool_path == '0':
            tool_path = misc.find_path(Project_path, 'sgs_calibration')
        if phase == 'Statistics':
            if os.path.exists('tensor_min_max.txt'):
                os.remove('tensor_min_max.txt')
            if os.path.exists('tensor_statistics.txt'):
                os.remove('tensor_statistics.txt')
            if os.path.exists('tensor_statistics_type.txt'):
                os.remove('tensor_statistics_type.txt')
            if os.path.exists('tensor_weight.txt'):
                os.remove('tensor_weight.txt')
            if os.path.exists('tensor_weight_calibration.txt'):
                os.remove('tensor_weight_calibration.txt')
            if os.path.exists('tensor_qab.txt'):
                os.remove('tensor_qab.txt')
            if os.path.exists('tensor_chn.txt'):
                os.remove('tensor_chn.txt')
            if os.path.exists('softmax_lut_range.txt'):
                os.remove('softmax_lut_range.txt')
            label = misc.Fake_Label(os.path.basename(model).split('.')[0])

            if args.quant_params_file is not None:
                use_qkp = False
                use_qio = False
                use_qki = False

            if use_qkp or use_qio:
                statistics.sgs_calibration_statistics('qkp', input_config, img_list, label.label_name, model, category, tool_path, model_name, preprocess_func,
                                                      debug_info, save_in)

            print('\033[31mStart to run statistics min/max...\033[0m')
            if num_subsets < 2:
                statistics.sgs_calibration_statistics('minmax', None, img_list, label.label_name, model, category, tool_path, model_name, preprocess_func, debug_info, save_in)
            else:
                statistics.run_statistics_minmax(img_list, label.label_name, model, category, model_name, preprocess_func, Project_path, tool_path,
                                                save_in, num_subsets=num_subsets)
            print('\033[31mRun Statistics Min Max OK.\033[0m')

            cmodel_float_model = model
            if use_qio or use_qkp:
                print('\033[31mStart to run statistics histogram...\033[0m')
                start_time = time.time()
                if num_subsets < 2:
                    statistics.sgs_calibration_statistics('histogram', None, img_list, label.label_name, model, category, tool_path, model_name, preprocess_func, debug_info, save_in)
                else:
                    statistics.run_statistics_kl(img_list, label.label_name, model, category, model_name, preprocess_func, Project_path, tool_path,
                                                    save_in, num_subsets=num_subsets)
                print('\033[31mRun Statistics histogram OK.\033[0m')

                if use_qio:
                    quantize_method = 'qio'
                else:
                    quantize_method = None
                print('\033[31mCalc {} and update minmax file...\033[0m'.format(quantize_method))
                quantize(tensors_file ='./tensor_weight.txt', by_channel=use_qkp, tensor_statistics_file='./tensor_statistics.txt', quantize_method=quantize_method,
                            minmax_file='./tensor_min_max.txt', parallel=num_subsets, verbose=False, suggest_type=use_qki, sth=sth,
                            ib_strategy=ib_strategy, kb_strategy=kb_strategy)
                stop_time = time.time()
                print('\033[31mCalc {} and update minmax file OK. Cost time: {}.\033[0m'.format(quantize_method,
                        time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time))))

            if args.quant_params_file is not None:
                quantize(tensors_file='./tensor_weight.txt', by_channel=use_qkp, tensor_statistics_file='./tensor_statistics.txt',
                            minmax_file='./tensor_min_max.txt', parallel=num_subsets, verbose=False, suggest_type=use_qki, sth=sth,
                            ib_strategy=ib_strategy, kb_strategy=kb_strategy, quant_params_file=args.quant_params_file)

            # merge minmax and weight to model
            if debug_info == 'convert_cmodel_float':
                debug_cmodel_float = True
            else:
                debug_cmodel_float = False

            m_p = 'tensor_min_max.txt'
            o_p = 'tensor_chn.txt'
            s_p = 'softmax_lut_range.txt'
            cmodel_float_model = statistics.sgs_calibration_fixed('Cmodel_float', model, tool_path, Dev_path, input_config, m_p, o_p, s_p, output, debug_cmodel_float)

        elif phase == 'Fixed':
            if (not re.search(r'_cmodel_float\.sim$', model, re.I)):
                raise NameError('Run Statistics phase first.')

            print('\033[31mStart to run convert fixed network...\033[0m')
            statistics.sgs_calibration_fixed('Fixed', model, tool_path, Dev_path, input_config, None, None, None, output, debug, quantThreshold=quant_threshold)
            print('\033[31mRun Fixed OK.\033[0m')

        elif phase == 'Fixed_without_IPU':
            if (not re.search(r'_cmodel_float\.sim$', model, re.I)):
                raise NameError('Run Statistics phase first.')

            print('\033[31mStart to run convert fixed network without ipu contorl...\033[0m')
            statistics.sgs_calibration_fixed('Fixed_without_IPU', model, tool_path, Dev_path, input_config, None, None, None, output, debug, quantThreshold=quant_threshold)
            print('\033[31mRun Fixed without IPU OK.\033[0m')

        else:
            raise NameError('Indicate calibration phase first.')
    else:
        raise OSError('Run `source cfg_env.sh` in top directory.')


if __name__ == '__main__':
    main()
