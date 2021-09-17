# -*- coding: utf-8 -*-

import os
from utils import misc
from utils import accuracy
import numpy as np


def set_input(data, save_name):
    list_data = list(data.flat)
    hw_str = '{}, {}'.format(1, 4096)
    with open(save_name, 'w') as f:
        f.write('Input_shape=%s\n' % hw_str)
        for num, i in enumerate(list_data):
            f.write('{}, '.format(i))
            if (num + 1) % 16 == 0:
                f.write('\n')


def parse_output(model, phase, tensor_names):
    output_dir = 'output'
    out_list = [f for f in os.listdir(output_dir) if f.startswith('unknown_{}'.format(model.split('/')[-1])) and f.endswith('.txt')]
    out_file = os.path.join(output_dir, out_list[0])
    results = dict()
    if phase == 'Float':
        with open(out_file, 'r') as f:
            start_read = False
            find_tensor = False
            for line in f:
                m = [i in line for i in tensor_names]
                for i in range(len(m)):
                    if m[i] == True:
                        tensor_name = tensor_names[i]
                        find_tensor = True
                        tensor_names.remove(tensor_name)
                if find_tensor:
                    if 'tensor data:' in line:
                        start_read = True
                        data = []
                    elif '}' in line:
                        start_read = False
                        find_tensor = False
                        data = np.array(data)
                        results[tensor_name] = data
                    elif start_read:
                        s = line.split('  ')[:-1]
                        data.extend([float(d) for d in s])
    else:
        with open(out_file, 'r') as f:
            start_read = False
            find_tensor = False
            for line in f:
                m = [i in line for i in tensor_names]
                for i in range(len(m)):
                    if m[i] == True:
                        tensor_name = tensor_names[i]
                        find_tensor = True
                        tensor_names.remove(tensor_name)
                if find_tensor:
                    if 'dim:' in line:
                        align_s = line.split(',')[-1].split('[')[-1].split(']')[0].split(' ')
                        ori_s = line.split(',')[-2].split('[')[-1].split(']')[0].split(' ')
                        align_c = [int(i) for i in align_s]
                        ori_c = [int(i) for i in ori_s]
                    elif 'tensor data:' in line:
                        start_read = True
                        data = []
                    elif '}' in line:
                        start_read = False
                        find_tensor = False
                        data = np.array(data)
                        data = np.reshape(data, align_c[:])
                        data = data[:, :ori_c[-1]]
                        results[tensor_name] = data
                    elif start_read:
                        s = line.split('  ')[:-1]
                        data.extend([float(d) for d in s])

    return results


def init_input(shape, value):
    data = np.ones(shape, dtype=np.int) * value
    return data


def read_date(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(float(line.strip()))

    return np.array(data)


def lstm_cmd(simulator, input_data, model, label, phase, debug):
    if debug:
        debug_info = 'gdb --args '
        if phase == 'Offline':
            label_cmd = '{}{} -i {}:{}:{}:{} -m {} -l {} -c Unknown -d offline --skip_preprocess'.format(debug_info, 
                        simulator, input_data[0], input_data[1], input_data[2], input_data[3], model, label)
        else:
            label_cmd = '{}{} -i {}:{}:{}:{} -m {} -l {} -c Unknown --skip_preprocess'.format(debug_info, 
                        simulator, input_data[0], input_data[1], input_data[2], input_data[3], model, label)
    else:
        if phase == 'Offline':
            label_cmd = '{} -i {}:{}:{}:{} -m {} -l {} -c Unknown -d offline --skip_preprocess  > lstm.log'.format(
                        simulator, input_data[0], input_data[1], input_data[2], input_data[3], model, label)
        else:
            label_cmd = '{} -i {}:{}:{}:{} -m {} -l {} -c Unknown --skip_preprocess > lstm.log'.format(
                        simulator, input_data[0], input_data[1], input_data[2], input_data[3], model, label)
    return label_cmd


def run_lstm(input_data, model, label, model_name, Project_path, simulator, phase, debug=False):
    output_dir = 'output'
    if model_name == '0':
        label_cmd = lstm_cmd(simulator, input_data, model, label, phase, debug)
        # print(label_cmd)
        os.system(label_cmd)
        out_list = [f for f in os.listdir(output_dir) if f.startswith('unknown_{}'.format(model.split('/')[-1])) and f.endswith('.txt')]
        out_result = os.path.join(output_dir, out_list[0])
        if not os.path.exists(out_result):
            raise RuntimeError('Run lstm_simulator failed!\nUse command to debug: {}'.format(label_cmd))


def run_lstm_calibrator(signal, input_data, model, label, model_name, Project_path, calibrator, phase):
    if signal == 'minmax':
        if model_name == '0':
            calibrator_cmd = '{} -i {}:{}:{}:{} -l {} -m {} -s min_max -p statistics -c Unknown --skip_preprocess > lstm_calibrator.log'.format(calibrator, input_data[0], 
                    input_data[1], input_data[2], input_data[3], label, model)
            os.system(calibrator_cmd)
            out_result = 'tensor_min_max.txt'
            if not os.path.exists(out_result):
                raise RuntimeError('Run lstm_calibrator failed!\nUse command to debug: {}'.format(calibrator_cmd))
    elif signal == 'qab':
        if model_name == '0':
            calibrator_cmd = '{} -i {}:{}:{}:{} -l {} -m {} -s qab -p statistics -c Unknown --skip_preprocess > lstm_calibrator.log'.format(calibrator, input_data[0], 
                    input_data[1], input_data[2], input_data[3], label, model)
            os.system(calibrator_cmd)
            out_result = 'tensor_qab.txt'
            if not os.path.exists(out_result):
                raise RuntimeError('Run lstm_calibrator failed!\nUse command to debug: {}'.format(calibrator_cmd))
