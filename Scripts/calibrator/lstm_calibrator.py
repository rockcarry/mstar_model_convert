# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from utils import statistics
from utils import misc
from utils import test_lstm

def arg_parse():
    parser = argparse.ArgumentParser(description='Simulator Tool')
    parser.add_argument('-i', '--inputs', type=str, required=True,
                        help='Directory containing inputs path.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    parser.add_argument('-u', '--units', type=int, required=True,
                        help='Number of LSTM units.')
    parser.add_argument('-t', '--type', type=str, required=True,
                        choices=['Float', 'Fixed', 'Offline'],
                        help='Indicate model data type.')
    parser.add_argument('-n', '--model_name', type=str, default='0',
                        help='Name of model to select image preprocess method')
    parser.add_argument('--tool', default='0', type=str, help='sgs_calibrator path.')

    return parser.parse_args()



def main():
    args = arg_parse()
    inputs_path = args.inputs
    model = args.model
    input_config = args.input_config
    units = args.units
    phase = args.type
    model_name = args.model_name
    tool = args.tool

    inputs_list = []
    try:
        inputs_list = os.listdir(inputs_path)
        inputs_list.sort(key=lambda x: int(x[:-4]))
        inputs_list = [os.path.join(os.path.abspath(inputs_path), im) for im in inputs_list]
    except:
        if os.path.exists(inputs_path):
            inputs_list.append(os.path.abspath(inputs_path))
        else:
            raise FileNotFoundError('No such {} image or directory.'.format(inputs_path))

    assert os.path.exists(model)
    model = os.path.abspath(model)

    if 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        Dev_path = os.path.join(Project_path, 'cfg')
        if tool == '0':
            calibrator = misc.find_path(Project_path, 'sgs_calibrator')
        else:
            calibrator = tool
    elif 'TOP_DIR' in os.environ:
        Project_path = os.environ['TOP_DIR']
        Dev_path = os.path.join(Project_path, '../SRC/Tool/cfg')
        if tool == '0':
            calibrator = misc.find_path(Project_path, 'sgs_calibration')
        else:
            calibrator = tool
    else:
        raise OSError('Run source cfg_env.sh in top directory.')

    label_txt = misc.fake_label(model_name)
    clip_txt = os.path.join(os.path.join('tmp_image', 'clip_txt'))
    data_txt = os.path.join(os.path.join('tmp_image', 'data_txt'))
    c_n_txt = os.path.join(os.path.join('tmp_image', 'c_n_txt'))
    h_n_txt = os.path.join(os.path.join('tmp_image', 'h_n_txt'))
    misc.renew_folder('output')
    misc.renew_folder('tmp_image')
    data_size = len(inputs_list)
    cycles = data_size // units
    left_size = data_size % units
    if left_size > 0:
        cycles += 1
    process_bar1 = misc.ShowProcess(data_size)
    process_bar2 = misc.ShowProcess(data_size)
    print('Phase 1')
    for index in range(cycles):
        id_start = index * units
        id_end = id_start + units
        if id_end > data_size:
            id_end = data_size
        clip_data = test_lstm.init_input((1, 256), 1)
        c_n_data = test_lstm.init_input((1, 256), 0)
        h_n_data = test_lstm.init_input((1, 256), 0)
        test_lstm.set_input(clip_data, clip_txt)
        test_lstm.set_input(c_n_data, c_n_txt)
        test_lstm.set_input(h_n_data, h_n_txt)
        for input_num in range(id_start, id_end):
            data_data = test_lstm.read_date(inputs_list[input_num])
            test_lstm.set_input(data_data, data_txt)
            test_lstm.run_lstm_calibrator('minmax', [data_txt, clip_txt, c_n_txt, h_n_txt], model, label_txt, model_name, Project_path, calibrator, phase)
            results = test_lstm.parse_output('statistics', phase, ['c_n', 'h_n', 'probs'])
            test_lstm.set_input(results['c_n'], c_n_txt)
            test_lstm.set_input(results['h_n'], h_n_txt)
            process_bar1.show_process()
    print('Phase 2')
    for index in range(cycles):
        id_start = index * units
        id_end = id_start + units
        if id_end > data_size:
            id_end = data_size
        clip_data = test_lstm.init_input((1, 256), 1)
        c_n_data = test_lstm.init_input((1, 256), 0)
        h_n_data = test_lstm.init_input((1, 256), 0)
        test_lstm.set_input(clip_data, clip_txt)
        test_lstm.set_input(c_n_data, c_n_txt)
        test_lstm.set_input(h_n_data, h_n_txt)
        for input_num in range(id_start, id_end):
            data_data = test_lstm.read_date(inputs_list[input_num])
            test_lstm.set_input(data_data, data_txt)
            test_lstm.run_lstm_calibrator('qab', [data_txt, clip_txt, c_n_txt, h_n_txt], model, label_txt, model_name, Project_path, calibrator, phase)
            results = test_lstm.parse_output('statistics', phase, ['c_n', 'h_n', 'probs'])
            test_lstm.set_input(results['c_n'], c_n_txt)
            test_lstm.set_input(results['h_n'], h_n_txt)
            process_bar2.show_process()
    print('Phase 3')
    statistics.sgs_calibration_fixed(True, model, calibrator, Dev_path, input_config, 'tensor_min_max.txt', 'tensor_qab.txt', 'softmax_lut_range.txt')






if __name__ == "__main__":
    main()