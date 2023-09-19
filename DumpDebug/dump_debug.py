#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import struct
import sys
from collections import OrderedDict
from functools import reduce


def cos_similarity(vector1, vector2):
    vec1 = np.asarray(vector1).flatten()
    vec2 = np.asarray(vector2).flatten()
    len1 = np.sqrt(np.dot(vec1, vec1))
    len2 = np.sqrt(np.dot(vec2, vec2))
    if len1 < 1e-8 or len2 < 1e-8:
        return 0
    else:
        cor = np.dot(vec1, vec2)
        return cor / (len1 * len2)


def calculate_mse(vector1, vector2):
    vec1 = np.asarray(vector1).flatten()
    vec2 = np.asarray(vector2).flatten()
    diff = np.square(np.abs(vec1 - vec2))
    diffSum = np.sum(diff)
    return diffSum / vec2.size


def calculate_rmse(vector1, vector2):
    vec1 = np.asarray(vector1).flatten()
    vec2 = np.asarray(vector2).flatten()
    rSum = np.sum(np.abs(vec2))
    diff = np.abs(vec1 - vec2)
    rdiffSum = np.sum(diff)
    return rdiffSum / rSum


def parse_dump_bin(bin_path):
    dump_dict = OrderedDict()
    variable_dict = set()
    with open(bin_path, 'rb') as f:
        name = ''
        shape = []
        for data in f:
            if b'name:' in data and b'shape:' in data:
                shape_str = data.decode('ascii').split('shape:[')[-1].strip().split('] dims:')[0]
                shape = [int(i) for i in shape_str.split(',')]
                name = data.decode('ascii').split('name: ')[-1].strip().split(' bConstant:')[0]
                if b'bConstant:0' in data:
                    variable_dict.add(name)
            if b'buffer data size:' in data:
                data_size = int(data.decode('ascii').split('size:')[-1].strip())
                float_data_bin = f.read(data_size)
                float_data = struct.unpack('f' * (data_size // 4), float_data_bin)
                shape_size = reduce(lambda x, y: x * y, shape)
                float_data = float_data[:shape_size]
                float_data_np = np.array(float_data)
                dump_dict[name] = float_data_np
                name = ''
                shape = []
                addition_read_len = len('\n}};\n//buffer data size: {}\n'.format(data_size))
                f.read(addition_read_len)
    return dump_dict, variable_dict


def write_txt(f, data):
    data_list = data.flatten().tolist()
    for num, value in enumerate(data_list):
        f.write('{:.6f}  '.format(value))
        if (num + 1) % 16 == 0:
            f.write('\n')
    if len(data_list) % 16 != 0:
        f.write('\n')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('python3', sys.argv[0], '[fixed.bin] [float.bin] (\"[tensor_name]\")')
        sys.exit(1)

    fixed_data, variable_tensors = parse_dump_bin(sys.argv[1])
    float_data, _ = parse_dump_bin(sys.argv[2])

    if len(sys.argv) == 3:
        names = []
        for name in fixed_data.keys():
            if name in float_data.keys() and name in variable_tensors:
                names.append(name)
        for idx, name in enumerate(names):
            try:
                cos = cos_similarity(fixed_data[name], float_data[name])
                rmse = calculate_rmse(fixed_data[name], float_data[name])
                mse = calculate_mse(fixed_data[name], float_data[name])
                print('{}.{}\tMSE:\t{:.6f}\tCOS:\t{:.6f}\tRMSE:\t{:.6f}'.format(idx, name, mse, cos, rmse))
            except ValueError as e:
                continue
    elif len(sys.argv) == 4:
        name = sys.argv[3]
        with open('fixed.txt', 'w') as f:
            f.write('{}\n'.format(name))
            write_txt(f, fixed_data[name])
        with open('float.txt', 'w') as f:
            f.write('{}\n'.format(name))
            write_txt(f, float_data[name])
        print('Data saved in fixed.txt and float.txt')