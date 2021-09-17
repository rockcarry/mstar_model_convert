# -*- coding: utf-8 -*-

import os
import re
import cv2
import shutil
import subprocess
from . import misc
import numpy as np
from multiprocessing import Pool
import pdb
import time

def statistics_minmax_multi(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path, save_in=False,
                            num_subsets=10, subset_index=0):
    statistics_folder = 'statistics'
    dataset_size = len(image_list)
    subset_size = list(((dataset_size // num_subsets) * np.ones((1, num_subsets), dtype=np.int)).flat)
    left_size = dataset_size % num_subsets
    if (left_size != 0):
        for i_ in range(len(subset_size)):
            if (i_ // left_size == 0):
                subset_size[i_] += 1
    if subset_index == 0:
        process_bar = misc.ShowProcess(subset_size[0])

    minmax = []
    if num_subsets == subset_index:
        tensor_minmax = 'tensor_min_max.txt'
        softmax_lut = 'softmax_lut_range.txt'
        tensor_chn = 'tensor_chn.txt'
        sub_dir0_minmax_txt = os.path.join(statistics_folder, 'sub_{:05d}'.format(0), 'tensor_min_max.txt')
        with open(sub_dir0_minmax_txt, 'r') as fs:
            minmax = fs.readlines()
            minmax = [i.strip().split(', ') for i in minmax]
            minmax = [[j.split(';')[0] for j in i] for i in minmax]
        minmax_len = len(minmax)
        for index in range(1, num_subsets):
            sub_dir_minmax = os.path.join(statistics_folder, 'sub_{:05d}'.format(index), 'tensor_min_max.txt')
            with open(sub_dir_minmax, 'r') as f:
                lines = f.readlines()
                lines = [i.strip().split(', ') for i in lines]
                lines = [[j.split(';')[0] for j in i] for i in lines]
            for row in range(minmax_len):
                try:
                    minmax_row = minmax[row]
                    lines_row = lines[row]
                except:
                    print('[sub_{:05d}][{}] tensor_min_max not in same row'.format(index, row))
                    minmax_len = row
                    break
                if minmax_row[0] == lines_row[0]:
                    min_0 = float(minmax_row[1].split(': ')[1])
                    max_0 = float(minmax_row[2].split(': ')[1])
                    min_i = float(lines_row[1].split(': ')[1])
                    max_i = float(lines_row[2].split(': ')[1])
                    if min_i < min_0:
                        min_0 = min_i
                    if max_i > max_0:
                        max_0 = max_i
                    minmax[row] = [minmax_row[0], 'min: {:06f}'.format(min_0), 'max: {:06f}'.format(max_0)]
                else:
                    print('[sub_{:05d}][{}] tensor_min_max not in same tensor name'.format(index, row))
                    minmax_len = row
                    break
        with open(tensor_minmax, 'w') as f:
            for i in minmax:
                contents = '{}, {}, {};\n'.format(i[0], i[1], i[2])
                f.write(contents)

        softmax = []
        sub_dir0_softmax_txt = os.path.join(statistics_folder, 'sub_{:05d}'.format(0), 'softmax_lut_range.txt')
        with open(sub_dir0_softmax_txt, 'r') as fs:
            softmax = fs.readlines()
            softmax = [i.strip().split(', ') for i in softmax]
            softmax = [[j.split(';')[0] for j in i] for i in softmax]
        softmax_len = len(softmax)
        for index in range(1, num_subsets):
            sub_dir_softmax = os.path.join(statistics_folder, 'sub_{:05d}'.format(index), 'softmax_lut_range.txt')
            with open(sub_dir_softmax, 'r') as f:
                lines = f.readlines()
                lines = [i.strip().split(', ') for i in lines]
                lines = [[j.split(';')[0] for j in i] for i in lines]
            for row in range(softmax_len):
                try:
                    softmax_row = softmax[row]
                    lines_row = lines[row]
                except:
                    print('[sub_{:05d}][{}] softmax_lut_range not in same row'.format(index, row + 1))
                    softmax_len = row
                    break
                if softmax_row[0] == lines_row[0]:
                    min_0 = float(softmax_row[1].split(': ')[1])
                    max_0 = float(softmax_row[2].split(': ')[1])
                    min_i = float(lines_row[1].split(': ')[1])
                    max_i = float(lines_row[2].split(': ')[1])
                    if min_i < min_0:
                        min_0 = min_i
                    if max_i > max_0:
                        max_0 = max_i
                    softmax[row] = [softmax_row[0], 'min: {:06f}'.format(min_0), 'max: {:06f}'.format(max_0)]
                else:
                    print('[sub_{:05d}][{}] softmax_lut_range not in same tensor name'.format(index, row + 1))
                    softmax_len = row
                    break
        with open(softmax_lut, 'w') as f:
            for i in softmax:
                contents = '{}, {}, {};\n'.format(i[0], i[1], i[2])
                f.write(contents)

        ob = []
        sub_dir0_ob_txt = os.path.join(statistics_folder, 'sub_{:05d}'.format(0), 'tensor_chn.txt')
        with open(sub_dir0_ob_txt, 'r') as fs:
            ob = fs.read()
            ob = ob.strip().split(';')[:-1]
            ob = [i.strip().split('\n') for i in ob]
            ob = [i[0] + ', ' + i[1] for i in ob]

        ob_len = len(ob)
        for index in range(1, num_subsets):
            sub_dir_ob = os.path.join(statistics_folder, 'sub_{:05d}'.format(index), 'tensor_chn.txt')
            with open(sub_dir_ob, 'r') as f:
                lines = f.read()
                lines = lines.strip().split(';')[:-1]
                lines = [i.strip().split('\n') for i in lines]
                lines = [i[0] + ', ' + i[1] for i in lines]
            for row in range(ob_len):
                try:
                    ob_row = ob[row]
                    lines_row = lines[row]
                except:
                    print('[sub_{:05d}][{}] tensor_chn not in same row'.format(index, row + 1))
                    ob_len = row
                    break
                if ob_row.split(', ')[0] == lines_row.split(', ')[0]:
                    num_ob = int(ob_row.split(', ')[1])
                    front_num = re.search(r'min_chn: ', ob_row, re.I).start()
                    min_chn0_start = re.search(r'min_chn: ', ob_row, re.I).end()
                    min_chn0_end = re.search(r'max_chn: ', ob_row, re.I).start() - 1
                    max_chn0_start = re.search(r'max_chn: ', ob_row, re.I).end()
                    max_chn0_end = len(ob_row)
                    min_chni_start = re.search(r'min_chn: ', lines_row, re.I).end()
                    min_chni_end = re.search(r'max_chn: ', lines_row, re.I).start() - 1
                    max_chni_start = re.search(r'max_chn: ', lines_row, re.I).end()
                    max_chni_end = len(lines_row)
                    minq_0 = np.array([float(i) for i in ob_row[min_chn0_start: min_chn0_end].split(',')[:num_ob]])
                    minq_i = np.array([float(i) for i in lines_row[min_chni_start: min_chni_end].split(',')[:num_ob]])
                    maxq_0 = np.array([float(i) for i in ob_row[max_chn0_start: max_chn0_end].split(',')[:num_ob]])
                    maxq_i = np.array([float(i) for i in lines_row[max_chni_start: max_chni_end].split(',')[:num_ob]])
                    minq = np.vstack([minq_0, minq_i]).min(0)
                    maxq = np.vstack([maxq_0, maxq_i]).max(0)
                    str_minq = '{}'.format(list(minq))
                    str_minq = str_minq[1: -1]
                    str_maxq = '{}'.format(list(maxq))
                    str_maxq = str_maxq[1: -1]
                    ob[row] = ob_row[:front_num] + 'min_chn: {}'.format(str_minq) + ', max_chn: {}'.format(str_maxq)
                else:
                    print('[sub_{:05d}][{}] tensor_chn not in same tensor name'.format(index, row + 1))
                    ob_len = row
                    break
        with open(tensor_chn, 'w') as f:
            for i in ob:
                front_num = re.search(r'min_chn: ', i, re.I).start()
                contents = i[:front_num - 2] + '\n' + i[front_num:] + ';\n'
                f.write(contents)

    else:
        sub_dir = os.path.join(statistics_folder, 'sub_{:05d}'.format(subset_index))
        os.mkdir(sub_dir)
        id_start = 0
        if subset_index > 0:
            for i_, size_ in enumerate(subset_size):
                if i_ < subset_index:
                    id_start += size_
        id_end = id_start + subset_size[subset_index]
        os.chdir(sub_dir)
        for i in range(id_start, id_end):
            im_file = image_list[i]
            if model_name != '0':
                if len(preprocess_func) > 1:
                    if len(preprocess_func) != len(im_file):
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = ''
                    for idx, preprocess in enumerate(preprocess_func):
                        out_img = '{}_{}.{}.data'.format(idx, os.path.basename(im_file[idx]), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file[idx], out_img, preprocess)
                        output_img += out_img
                        if idx < len(preprocess_func) - 1:
                            output_img += ':'
                else:
                    if isinstance(im_file, list):
                        if len(preprocess_func) != 1 or len(im_file) != 1:
                            raise ValueError('Got different num of preprocess_methods and images!')
                        output_img = '{}.{}.data'.format(os.path.basename(im_file[0]), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file[0], output_img, preprocess_func[0])
                    else:
                        output_img = '{}.{}.data'.format(os.path.basename(im_file), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file, output_img, preprocess_func[0])
                sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s min_max -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                    tool_path, output_img, label, model, category, model.split('.')[-2].split('/')[-1]
                )
                os.system(sgs_calibration_minmax)
                if subset_index == 0:
                    process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt')):
                    raise RuntimeError('Run Statistics MinMax failed!\nUse command to debug: {}'.format(sgs_calibration_minmax))
                if not save_in:
                    for out_img in output_img.split(':'):
                        os.remove(out_img)
            else:
                sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s min_max -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                    tool_path, im_file, label, model, category, model.split('.')[-2].split('/')[-1]
                )
                os.system(sgs_calibration_minmax)
                if subset_index == 0:
                    process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt')):
                    raise RuntimeError('Run Statistics MinMax failed!\nUse command to debug: {}'.format(sgs_calibration_minmax))


def statistics_kl(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path, save_in=False,
                            num_subsets=10, subset_index=0):
    statistics_folder = 'statistics'
    dataset_size = len(image_list)
    subset_size = list(((dataset_size // num_subsets) * np.ones((1, num_subsets), dtype=np.int)).flat)
    left_size = dataset_size % num_subsets
    if (left_size != 0):
        for i_ in range(len(subset_size)):
            if (i_ // left_size == 0):
                subset_size[i_] += 1
    if subset_index == 0:
        process_bar = misc.ShowProcess(subset_size[0])

    kl_data = []
    if num_subsets == subset_index:
        tensor_statistics = 'tensor_statistics.txt'
        sub_dir0_kl_txt = os.path.join(statistics_folder, 'sub_{:05d}'.format(0), 'tensor_statistics.txt')
        with open(sub_dir0_kl_txt, 'r') as fs:
            kl_data = fs.readlines()
            kl_data = [i.strip().split('; ') for i in kl_data]
            kl_data = [[j.split(';')[0] for j in i] for i in kl_data]
        kl_len = len(kl_data)
        for index in range(1, num_subsets):
            sub_dir_kl = os.path.join(statistics_folder, 'sub_{:05d}'.format(index), 'tensor_statistics.txt')
            with open(sub_dir_kl, 'r') as f:
                lines = f.readlines()
                lines = [i.strip().split('; ') for i in lines]
                lines = [[j.split(';')[0] for j in i] for i in lines]
            for row in range(kl_len):
                try:
                    kl_row = kl_data[row]
                    lines_row = lines[row]
                except:
                    print('[sub_{:05d}][{}] tensor_statistics not in same row'.format(index, row))
                    kl_len = row
                    break
                if kl_row[0] == lines_row[0]:
                    kl_0 = np.array([int(i) for i in kl_row[-1].split(': ')[-1].split(', ') if i != ''], dtype=np.int)
                    kl_i = np.array([int(i) for i in lines_row[-1].split(': ')[-1].split(', ') if i != ''], dtype=np.int)
                    kl_0 = np.add(kl_0, kl_i)
                    str_kl_0 = '{}'.format(list(kl_0))
                    str_kl_0 = str_kl_0[1:-1]
                    kl_data[row] = [kl_row[0], kl_row[1], kl_row[2], kl_row[3], kl_row[4], kl_row[5], kl_row[6],kl_row[7], 'KLHistogram: {}'.format(str_kl_0)]
                else:
                    print('[sub_{:05d}][{}] tensor_statistics not in same tensor name'.format(index, row))
                    kl_len = row
                    break
        with open(tensor_statistics, 'w') as f:
            for i in kl_data:
                contents = '{}; {}; {}; {}; {}; {}; {}; {}; {};\n'.format(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8])
                f.write(contents)

    else:
        sub_dir = os.path.join(statistics_folder, 'sub_{:05d}'.format(subset_index))
        if not (os.path.exists(sub_dir) and os.path.exists(os.path.join(sub_dir, 'tensor_min_max.txt'))):
            raise FileNotFoundError('No tensor file or directory.')
        id_start = 0
        if subset_index > 0:
            for i_, size_ in enumerate(subset_size):
                if i_ < subset_index:
                    id_start += size_
        id_end = id_start + subset_size[subset_index]
        os.chdir(sub_dir)
        for i in range(id_start, id_end):
            im_file = image_list[i]
            if model_name != '0':
                if len(preprocess_func) > 1:
                    if len(preprocess_func) != len(im_file):
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = ''
                    for idx, preprocess in enumerate(preprocess_func):
                        out_img = '{}_{}.{}.data'.format(idx, os.path.basename(im_file[idx]), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file[idx], out_img, preprocess)
                        output_img += out_img
                        if idx < len(preprocess_func) - 1:
                            output_img += ':'
                else:
                    if isinstance(im_file, list):
                        if len(preprocess_func) != 1 or len(im_file) != 1:
                            raise ValueError('Got different num of preprocess_methods and images!')
                        output_img = '{}.{}.data'.format(os.path.basename(im_file[0]), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file[0], output_img, preprocess_func[0])
                    else:
                        output_img = '{}.{}.data'.format(os.path.basename(im_file), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file, output_img, preprocess_func[0])
                sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s histogram -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                    tool_path, output_img, label, model, category, model.split('.')[-2].split('/')[-1]
                )
                os.system(sgs_calibration_minmax)
                if subset_index == 0:
                    process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt') and os.path.exists('tensor_statistics.txt')):
                    raise RuntimeError('Run Statistics MinMax failed!\nUse command to debug: {}'.format(sgs_calibration_minmax))
                if not save_in:
                    for out_img in output_img.split(':'):
                        os.remove(out_img)
            else:
                sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s histogram -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                    tool_path, im_file, label, model, category, model.split('.')[-2].split('/')[-1]
                )
                os.system(sgs_calibration_minmax)
                if subset_index == 0:
                    process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt') and os.path.exists('tensor_statistics.txt')):
                    raise RuntimeError('Run Statistics MinMax failed!\nUse command to debug: {}'.format(sgs_calibration_minmax))


def statistics_qab_multi(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path, save_in=False,
                         num_subsets=10, subset_index=0,quantThreshold=1024):
    statistics_folder = 'statistics'
    dataset_size = len(image_list)
    subset_size = list(((dataset_size // num_subsets) * np.ones((1, num_subsets), dtype=np.int)).flat)
    left_size = dataset_size % num_subsets
    if (left_size != 0):
        for i_ in range(len(subset_size)):
            if (i_ // left_size == 0):
                subset_size[i_] += 1
    if subset_index == 0:
        process_bar = misc.ShowProcess(subset_size[0])

    qab = []
    if num_subsets == subset_index:
        tensor_qab = 'tensor_qab.txt'
        softmax_lut = 'softmax_lut_range.txt'
        sub_dir0_qab_txt = os.path.join(statistics_folder, 'sub_{:05d}'.format(0), 'tensor_qab.txt')
        with open(sub_dir0_qab_txt, 'r') as fs:
            qab = fs.read()
            qab = qab.strip().split(';')[:-1]
            qab = [i.strip().split('\n') for i in qab]
            qab = [i[0] + ', ' + i[1] for i in qab]

        qab_len = len(qab)
        for index in range(1, num_subsets):
            sub_dir_qab = os.path.join(statistics_folder, 'sub_{:05d}'.format(index), 'tensor_qab.txt')
            with open(sub_dir_qab, 'r') as f:
                lines = f.read()
                lines = lines.strip().split(';')[:-1]
                lines = [i.strip().split('\n') for i in lines]
                lines = [i[0] + ', ' + i[1] for i in lines]
            for row in range(qab_len):
                try:
                    qab_row = qab[row]
                    lines_row = lines[row]
                except:
                    print('[sub_{:05d}][{}] tensor_qab not in same row'.format(index, row + 1))
                    qab_len = row
                    break
                if qab_row.split(', ')[0] == lines_row.split(', ')[0]:
                    num_qab = int(qab_row.split(', ')[1])
                    front_num = re.search(r'min_qab: ', qab_row, re.I).start()
                    min_qab0_start = re.search(r'min_qab: ', qab_row, re.I).end()
                    min_qab0_end = re.search(r'max_qab: ', qab_row, re.I).start() - 1
                    max_qab0_start = re.search(r'max_qab: ', qab_row, re.I).end()
                    max_qab0_end = len(qab_row)
                    min_qabi_start = re.search(r'min_qab: ', lines_row, re.I).end()
                    min_qabi_end = re.search(r'max_qab: ', lines_row, re.I).start() - 1
                    max_qabi_start = re.search(r'max_qab: ', lines_row, re.I).end()
                    max_qabi_end = len(lines_row)
                    minq_0 = np.array([int(i) for i in qab_row[min_qab0_start: min_qab0_end].split(',')[:num_qab]])
                    minq_i = np.array([int(i) for i in lines_row[min_qabi_start: min_qabi_end].split(',')[:num_qab]])
                    maxq_0 = np.array([int(i) for i in qab_row[max_qab0_start: max_qab0_end].split(',')[:num_qab]])
                    maxq_i = np.array([int(i) for i in lines_row[max_qabi_start: max_qabi_end].split(',')[:num_qab]])
                    minq = np.vstack([minq_0, minq_i]).min(0)
                    maxq = np.vstack([maxq_0, maxq_i]).max(0)
                    str_minq = '{}'.format(list(minq))
                    str_minq = str_minq[1: -1]
                    str_maxq = '{}'.format(list(maxq))
                    str_maxq = str_maxq[1: -1]
                    qab[row] = qab_row[:front_num] + 'min_qab: {}'.format(str_minq) + ', max_qab: {}'.format(str_maxq)
                else:
                    print('[sub_{:05d}][{}] tensor_qab not in same tensor name'.format(index, row + 1))
                    qab_len = row
                    break
        with open(tensor_qab, 'w') as f:
            for i in qab:
                front_num = re.search(r'min_qab: ', i, re.I).start()
                contents = i[:front_num - 2] + '\n' + i[front_num:] + ';\n'
                f.write(contents)

        softmax = []
        sub_dir0_softmax_txt = os.path.join(statistics_folder, 'sub_{:05d}'.format(0), 'softmax_lut_range.txt')
        with open(sub_dir0_softmax_txt, 'r') as fs:
            softmax = fs.readlines()
            softmax = [i.strip().split(', ') for i in softmax]
            softmax = [[j.split(';')[0] for j in i] for i in softmax]
        softmax_len = len(softmax)
        for index in range(1, num_subsets):
            sub_dir_softmax = os.path.join(statistics_folder, 'sub_{:05d}'.format(index), 'softmax_lut_range.txt')
            with open(sub_dir_softmax, 'r') as f:
                lines = f.readlines()
                lines = [i.strip().split(', ') for i in lines]
                lines = [[j.split(';')[0] for j in i] for i in lines]
            for row in range(softmax_len):
                try:
                    softmax_row = softmax[row]
                    lines_row = lines[row]
                except:
                    print('[sub_{:05d}][{}] softmax_lut_range not in same row'.format(index, row + 1))
                    softmax_len = row
                    break
                if softmax_row[0] == lines_row[0]:
                    min_0 = float(softmax_row[1].split(': ')[1])
                    max_0 = float(softmax_row[2].split(': ')[1])
                    min_i = float(lines_row[1].split(': ')[1])
                    max_i = float(lines_row[2].split(': ')[1])
                    if min_i < min_0:
                        min_0 = min_i
                    if max_i > max_0:
                        max_0 = max_i
                    softmax[row] = [softmax_row[0], 'min: {:06f}'.format(min_0), 'max: {:06f}'.format(max_0)]
                else:
                    print('[sub_{:05d}][{}] softmax_lut_range not in same tensor name'.format(index, row + 1))
                    softmax_len = row
                    break
        with open(softmax_lut, 'w') as f:
            for i in softmax:
                contents = '{}, {}, {};\n'.format(i[0], i[1], i[2])
                f.write(contents)

    else:
        sub_dir = os.path.join(statistics_folder, 'sub_{:05d}'.format(subset_index))
        if not (os.path.exists(sub_dir) and os.path.exists(os.path.join(sub_dir, 'tensor_min_max.txt'))):
            raise FileNotFoundError('No tensor file or directory.')
        id_start = 0
        if subset_index > 0:
            for i_, size_ in enumerate(subset_size):
                if i_ < subset_index:
                    id_start += size_
        id_end = id_start + subset_size[subset_index]
        os.chdir(sub_dir)
        for i in range(id_start, id_end):
            im_file = image_list[i]
            if model_name != '0':
                if len(preprocess_func) > 1:
                    if len(preprocess_func) != len(im_file):
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = ''
                    for idx, preprocess in enumerate(preprocess_func):
                        out_img = '{}_{}.{}.data'.format(idx, os.path.basename(im_file[idx]), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file[idx], out_img, preprocess)
                        output_img += out_img
                        if idx < len(preprocess_func) - 1:
                            output_img += ':'
                else:
                    if isinstance(im_file, list):
                        if len(preprocess_func) != 1 or len(im_file) != 1:
                            raise ValueError('Got different num of preprocess_methods and images!')
                        output_img = '{}.{}.data'.format(os.path.basename(im_file[0]), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file[0], output_img, preprocess_func[0])
                    else:
                        output_img = '{}.{}.data'.format(os.path.basename(im_file), os.path.basename(model).split('.')[0])
                        misc.convert_image(im_file, output_img, preprocess_func[0])
                sgs_calibration_qab = '{} -i \'{}\' -l {} -m {} -s qab --per_do_quant_threshold {} -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                    tool_path, output_img, label, model, quantThreshold, category, model.split('.')[-2].split('/')[-1]
                )
                os.system(sgs_calibration_qab)
                if subset_index == 0:
                    process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt') and (
                        os.path.exists('tensor_qab.txt'))):
                    raise RuntimeError('Run Statistics Qab failed!\nUse command to debug: {}'.format(sgs_calibration_qab))
                if not save_in:
                    for out_img in output_img.split(':'):
                        os.remove(out_img)
            else:
                sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s qab --per_do_quant_threshold {} -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                    tool_path, im_file, label, model, quantThreshold, category, model.split('.')[-2].split('/')[-1]
                )
                os.system(sgs_calibration_minmax)
                if subset_index == 0:
                    process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt') and (
                        os.path.exists('tensor_qab.txt'))):
                    raise RuntimeError('Run Statistics Qab failed!\nUse command to debug: {}'.format(sgs_calibration_qab))


def run_statistics_minmax(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path, save_in=False,
                          num_subsets=10):
    statistics_folder = 'statistics'
    if os.path.exists(statistics_folder):
        if os.path.isdir(statistics_folder):
            shutil.rmtree(statistics_folder)
            os.mkdir(statistics_folder)
        else:
            os.remove(statistics_folder)
            os.mkdir(statistics_folder)
    else:
        os.mkdir(statistics_folder)
    if len(image_list) < num_subsets:
        num_subsets = len(image_list)
    p_minmax = Pool(processes=num_subsets)
    for ip_mm in range(num_subsets):
        p_minmax.apply_async(statistics_minmax_multi, args=(image_list, label, model, category, model_name, preprocess_func,
                            project_path, tool_path), kwds={'save_in': save_in, 'num_subsets': num_subsets,
                            'subset_index': ip_mm})
    p_minmax.close()
    p_minmax.join()
    check_multi_statistics_after_sys(num_subsets, 'minmax', image_list, label, model, category, tool_path, model_name, preprocess_func)
    statistics_minmax_multi(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path,
                            save_in, num_subsets=num_subsets, subset_index=num_subsets)


def run_statistics_kl(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path, save_in=False,
                       num_subsets=10):
    tensor_minmax = 'tensor_min_max.txt'
    statistics_folder = 'statistics'
    if os.path.exists(tensor_minmax):
        for i in range(num_subsets):
            sub_dir = os.path.join(statistics_folder, 'sub_{:05d}'.format(i))
            shutil.copy(tensor_minmax, sub_dir)
    else:
        raise FileNotFoundError('Run Statistics_minmax first.')
    if len(image_list) < num_subsets:
        num_subsets = len(image_list)
    p_kl = Pool(processes=num_subsets)
    for ip_kl in range(num_subsets):
        p_kl.apply_async(statistics_kl, args=(image_list, label, model, category, model_name, preprocess_func,
                          project_path, tool_path), kwds={'save_in': save_in, 'num_subsets': num_subsets,
                          'subset_index': ip_kl})
    p_kl.close()
    p_kl.join()
    check_multi_statistics_after_sys(num_subsets, 'histogram', image_list, label, model, category, tool_path, model_name, preprocess_func)
    statistics_kl(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path,
                         save_in, num_subsets=num_subsets, subset_index=num_subsets)

def run_statistics_qab(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path, save_in=False,
                       num_subsets=10, quantThreshold=1024):
    tensor_minmax = 'tensor_min_max.txt'
    statistics_folder = 'statistics'
    if os.path.exists(tensor_minmax):
        for i in range(num_subsets):
            sub_dir = os.path.join(statistics_folder, 'sub_{:05d}'.format(i))
            shutil.copy(tensor_minmax, sub_dir)
    else:
        raise FileNotFoundError('Run Statistics_minmax first.')
    if len(image_list) < num_subsets:
        num_subsets = len(image_list)
    p_qab = Pool(processes=num_subsets)
    for ip_qab in range(num_subsets):
        p_qab.apply_async(statistics_qab_multi, args=(image_list, label, model, category, model_name, preprocess_func,
                        project_path, tool_path), kwds={'save_in': save_in, 'num_subsets': num_subsets,
                        'subset_index': ip_qab, 'quantThreshold': quantThreshold})
    p_qab.close()
    p_qab.join()
    check_multi_statistics_after_sys(num_subsets, 'qab', image_list, label, model, category, tool_path, model_name, preprocess_func)
    statistics_qab_multi(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path,
                         save_in, num_subsets=num_subsets, subset_index=num_subsets)


def sgs_calibration_statistics(signal, input_config, image_list, label, model, category, tool_path, model_name, preprocess_func, debug_info=False, save_in=False, log=False, quantThreshold=1024):
    if signal == 'minmax':
        process_bar = misc.ShowProcess(len(image_list))
        for image in image_list:
            if model_name != '0':
                if len(preprocess_func) > 1:
                    if len(preprocess_func) != len(image):
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = ''
                    for idx, preprocess in enumerate(preprocess_func):
                        out_img = '{}_{}.{}.data'.format(idx, os.path.basename(image[idx]), os.path.basename(model).split('.')[0])
                        misc.convert_image(image[idx], out_img, preprocess)
                        output_img += out_img
                        if idx < len(preprocess_func) - 1:
                            output_img += ':'
                else:
                    if isinstance(image, list):
                        if len(preprocess_func) != 1 or len(image) != 1:
                            raise ValueError('Got different num of preprocess_methods and images!')
                        output_img = '{}.{}.data'.format(os.path.basename(image[0]), os.path.basename(model).split('.')[0])
                        misc.convert_image(image[0], output_img, preprocess_func[0])
                    else:
                        output_img = '{}.{}.data'.format(os.path.basename(image), os.path.basename(model).split('.')[0])
                        misc.convert_image(image, output_img, preprocess_func[0])
                if debug_info == 'minmax':
                    sgs_calibration_minmax = 'gdb --args {} -i \'{}\' -l {} -m {} -s min_max -p statistics -c {} --skip_preprocess'.format(
                                            tool_path, output_img, label, model, category)
                    print('\033[33m================Debug command================\033[0m\n' + sgs_calibration_minmax + '\n\033[33m=============================================\033[0m')
                else:
                    if log:
                        sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s min_max -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                                                tool_path, output_img, label, model, category, model.split('.')[-2].split('/')[-1])
                    else:
                        sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s min_max -p statistics -c {} --skip_preprocess'.format(
                                                tool_path, output_img, label, model, category)
                os.system(sgs_calibration_minmax)
                process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt')):
                    raise RuntimeError('Run Statistics MinMax failed!\nUse command to debug: {}'.format(sgs_calibration_minmax))
                if not save_in:
                    for out_img in output_img.split(':'):
                        os.remove(out_img)
            else:
                if debug_info == 'minmax':
                    sgs_calibration_minmax = 'gdb --args {} -i \'{}\' -l {} -m {} -s min_max -p statistics -c {} --skip_preprocess'.format(
                                        tool_path, image, label, model, category)
                    print('\033[33m================Debug command================\033[0m\n' + sgs_calibration_minmax + '\n\033[33m=============================================\033[0m')
                else:
                    if log:
                        sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s min_max -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                                                tool_path, image, label, model, category, model.split('.')[-2].split('/')[-1])
                    else:
                        sgs_calibration_minmax = '{} -i \'{}\' -l {} -m {} -s min_max -p statistics -c {} --skip_preprocess'.format(
                                                tool_path, image, label, model, category)
                os.system(sgs_calibration_minmax)
                process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt')):
                    raise RuntimeError('Run Statistics MinMax failed!\nUse command to debug: {}'.format(sgs_calibration_minmax))
    elif signal == 'qab':
        process_bar = misc.ShowProcess(len(image_list))
        for image in image_list:
            if model_name != '0':
                if len(preprocess_func) > 1:
                    if len(preprocess_func) != len(image):
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = ''
                    for idx, preprocess in enumerate(preprocess_func):
                        out_img = '{}_{}.{}.data'.format(idx, os.path.basename(image[idx]), os.path.basename(model).split('.')[0])
                        misc.convert_image(image[idx], out_img, preprocess)
                        output_img += out_img
                        if idx < len(preprocess_func) - 1:
                            output_img += ':'
                else:
                    if isinstance(image, list):
                        if len(preprocess_func) != 1 or len(image) != 1:
                            raise ValueError('Got different num of preprocess_methods and images!')
                        output_img = '{}.{}.data'.format(os.path.basename(image[0]), os.path.basename(model).split('.')[0])
                        misc.convert_image(image[0], output_img, preprocess_func[0])
                    else:
                        output_img = '{}.{}.data'.format(os.path.basename(image), os.path.basename(model).split('.')[0])
                        misc.convert_image(image, output_img, preprocess_func[0])
                if debug_info == 'qab':
                    sgs_calibration_qab = 'gdb --args {} -i \'{}\' -l {} -m {} -s qab --per_do_quant_threshold {} -p statistics -c {} --skip_preprocess'.format(
                                        tool_path, output_img, label, model, quantThreshold, category)
                    print('\033[33m================Debug command================\033[0m\n' + sgs_calibration_qab + '\n\033[33m=============================================\033[0m')
                else:
                    if log:
                        sgs_calibration_qab = '{} -i \'{}\' -l {} -m {} -s qab --per_do_quant_threshold {} -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                                            tool_path, output_img, label, model, quantThreshold, category, model.split('.')[-2].split('/')[-1])
                    else:
                        sgs_calibration_qab = '{} -i \'{}\' -l {} -m {} -s qab --per_do_quant_threshold {} -p statistics -c {} --skip_preprocess'.format(
                                            tool_path, output_img, label, model, quantThreshold, category)
                os.system(sgs_calibration_qab)
                process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt') and (os.path.exists('tensor_qab.txt'))):
                    raise RuntimeError('Run Statistics Qab failed!\nUse command to debug: {}'.format(sgs_calibration_qab))
                if not save_in:
                    for out_img in output_img.split(':'):
                        os.remove(out_img)
            else:
                if debug_info == 'qab':
                    sgs_calibration_qab = 'gdb --args {} -i \'{}\' -l {} -m {} -s qab --per_do_quant_threshold {} -p statistics -c {} --skip_preprocess'.format(
                                        tool_path, image, label, model, quantThreshold, category)
                    print('\033[33m================Debug command================\033[0m\n' + sgs_calibration_qab + '\n\033[33m=============================================\033[0m')
                else:
                    if log:
                        sgs_calibration_qab = '{} -i \'{}\' -l {} -m {} -s qab --per_do_quant_threshold {} -p statistics -c {} --skip_preprocess>> {}_statistics.log'.format(
                                            tool_path, image, label, model, quantThreshold, category, model.split('.')[-2].split('/')[-1])
                    else:
                        sgs_calibration_qab = '{} -i \'{}\' -l {} -m {} -s qab --per_do_quant_threshold {} -p statistics -c {} --skip_preprocess'.format(
                                            tool_path, image, label, model, quantThreshold, category)
                os.system(sgs_calibration_qab)
                process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt') and (os.path.exists('tensor_qab.txt'))):
                    raise RuntimeError('Run Statistics Qab failed!\nUse command to debug: {}'.format(sgs_calibration_qab))
    elif signal == 'qkp':
        if debug_info == 'qkp':
            sgs_calibration_qkp = 'gdb --args {} -i \'{}\' -l {} --input_config {} -m {} -s qkp -p statistics -c {}'.format(
                                tool_path, image_list[0], label, input_config, model, category)
            print('\033[33m================Debug command================\033[0m\n' + sgs_calibration_qkp + '\n\033[33m=============================================\033[0m')
        else:
            sgs_calibration_qkp = '{} -i \'{}\' -l {} --input_config {} -m {} -s qkp -p statistics -c {} >> {}_statistics.log'.format(
                                tool_path, image_list[0], label, input_config, model, category, model.split('.')[-2].split('/')[-1])

        os.system(sgs_calibration_qkp)
        if not (os.path.exists('tensor_weight.txt')):
            raise RuntimeError('Run Statistics per channel quantization failed!\nUse command to debug: {}'.format(sgs_calibration_qkp))
    else:
        process_bar = misc.ShowProcess(len(image_list))
        for image in image_list:
            if model_name != '0':
                if len(preprocess_func) > 1:
                    if len(preprocess_func) != len(image):
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = ''
                    for idx, preprocess in enumerate(preprocess_func):
                        out_img = '{}_{}.{}.data'.format(idx, os.path.basename(image[idx]), os.path.basename(model).split('.')[0])
                        misc.convert_image(image[idx], out_img, preprocess)
                        output_img += out_img
                        if idx < len(preprocess_func) - 1:
                            output_img += ':'
                else:
                    if isinstance(image, list):
                        if len(preprocess_func) != 1 or len(image) != 1:
                            raise ValueError('Got different num of preprocess_methods and images!')
                        output_img = '{}.{}.data'.format(os.path.basename(image[0]), os.path.basename(model).split('.')[0])
                        misc.convert_image(image[0], output_img, preprocess_func[0])
                    else:
                        output_img = '{}.{}.data'.format(os.path.basename(image), os.path.basename(model).split('.')[0])
                        misc.convert_image(image, output_img, preprocess_func[0])
                if debug_info == 'histogram':
                    sgs_calibration_kl = 'gdb --args {} -i \'{}\' -l {} -m {} -s histogram -p statistics -c {} --skip_preprocess'.format(
                                        tool_path, output_img, label, model, category)
                    print('\033[33m================Debug command================\033[0m\n' + sgs_calibration_kl + '\n\033[33m=============================================\033[0m')
                else:
                    if log:
                        sgs_calibration_kl = '{} -i \'{}\' -l {} -m {} -s histogram -p statistics -c {} --skip_preprocess >> {}_statistics.log'.format(
                                            tool_path, output_img, label, model, category, model.split('.')[-2].split('/')[-1])
                    else:
                        sgs_calibration_kl = '{} -i \'{}\' -l {} -m {} -s histogram -p statistics -c {} --skip_preprocess'.format(
                                            tool_path, output_img, label, model, category)
                os.system(sgs_calibration_kl)
                process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt') and (os.path.exists('tensor_statistics.txt'))):
                    raise RuntimeError('Run Statistics histogram failed!\nUse command to debug: {}'.format(sgs_calibration_kl))
                if not save_in:
                    for out_img in output_img.split(':'):
                        os.remove(out_img)
            else:
                if debug_info == 'histogram':
                    sgs_calibration_kl = 'gdb --args {} -i \'{}\' -l {} -m {} -s histogram -p statistics -c {} --skip_preprocess'.format(
                                        tool_path, image, label, model, category)
                    print('\033[33m================Debug command================\033[0m\n' + sgs_calibration_kl + '\n\033[33m=============================================\033[0m')
                else:
                    if log:
                        sgs_calibration_kl = '{} -i \'{}\' -l {} -m {} -s histogram -p statistics -c {} --skip_preprocess>> {}_statistics.log'.format(
                                            tool_path, image, label, model, category, model.split('.')[-2].split('/')[-1])
                    else:
                        sgs_calibration_kl = '{} -i \'{}\' -l {} -m {} -s histogram -p statistics -c {} --skip_preprocess'.format(
                                            tool_path, image, label, model, category)
                os.system(sgs_calibration_kl)
                process_bar.show_process()
                if not (os.path.exists('tensor_min_max.txt') and os.path.exists('softmax_lut_range.txt') and (os.path.exists('tensor_statistics.txt'))):
                    raise RuntimeError('Run Statistics histogram failed!\nUse command to debug: {}'.format(sgs_calibration_kl))


def convert_to_unicode(output):
    if output is None:
        return u""

    if isinstance(output, bytes):
        try:
            return output.decode()
        except UnicodeDecodeError:
            pass
    return output


class ConverterError(Exception):
    pass


def PopenRun(cmdline, output_file):
    proc = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = proc.communicate()
    if not os.path.exists(output_file):
        stdout = convert_to_unicode(stdout)
        stderr = convert_to_unicode(stderr)
        raise ConverterError("Convert failed. See console for info.\n%s\n%s\n" % (stdout, stderr))


def sgs_calibration_fixed(signal, model, tool_path, dev_path, input_path, minmax_path, chn_path, softmax_path, output, debug=False, quantThreshold=1024):
    if signal == 'Fixed_without_IPU':
        compileconfig_path = misc.find_path(dev_path, 'Dev_tool_CompilerConfig.txt')
        if re.search(r'_cmodel_float\.sim$', model, re.I):
            out_model = model.replace('_cmodel_float.sim', '_fixed_without_ipu_ctrl.sim')
        elif re.search(r'_float\.sim$', model, re.I):
            out_model = model.replace('_float.sim', '_fixed_without_ipu_ctrl.sim')
        else:
            out_model = model.replace('.sim', '_fixed_without_ipu_ctrl.sim')
        if output is not None:
            if os.path.isdir(output):
                out_model = os.path.join(output, os.path.basename(out_model))
            elif (os.path.isdir(os.path.dirname(output)) or (os.path.dirname(output) == '')):
                out_model = output
        if debug:
            calibration_without_IPU = 'gdb --args {} -m {} -o {} -t {} --input_config {} --per_do_quant_threshold {} -p convert_fixed_without_ipu_ctrl'.format(
                tool_path, model, out_model, compileconfig_path, input_path, quantThreshold)
            print('\033[33m================Debug command================\033[0m\n' + calibration_without_IPU + '\n\033[33m=============================================\033[0m')
            os.system(calibration_without_IPU)
        else:
            calibration_without_IPU = '{} -m {} -o {} -t {} --input_config {} --per_do_quant_threshold {} -p convert_fixed_without_ipu_ctrl'.format(
                tool_path, model, out_model, compileconfig_path, input_path, quantThreshold)
            PopenRun(calibration_without_IPU, out_model)
        if not os.path.exists(out_model):
            raise RuntimeError('Run Calibration Fixed without IPU ctrl failed!\nUse command to debug: {}'.format(calibration_without_IPU))
        print('\033[31mFixed_without_IPU model at: {}\033[0m'.format(out_model))

    elif signal == 'Cmodel_float':
        if re.search(r'_cmodel_float\.sim$', model, re.I):
            out_model = model
        elif re.search(r'_float\.sim$', model, re.I):
            out_model = model.replace('_float.sim', '_cmodel_float.sim')
        else:
            out_model = model.replace('.sim', '_cmodel_float.sim')
        w_file = './tensor_weight_calibration.txt'
        infect_datatype = misc.Infect_Config('./infectDataTypeConfig.txt')
        if minmax_path == None:
            w_file = None
            infect_datatype.blank()
        else:
            infect_datatype.datatype()
        if debug:
            calibration_cmodel_float = 'gdb --args {} -m {}  -w {} -d {} -q {} -e {} -t {} -o {} --input_config {} -p convert_cmodel_float'.format(
                                tool_path, model, w_file, minmax_path, chn_path, softmax_path, infect_datatype.file_path,
                                out_model, input_path)
            print('\033[33m================Debug command================\033[0m\n' + calibration_cmodel_float + '\n\033[33m=============================================\033[0m')
            os.system(calibration_cmodel_float)
        else:
            calibration_cmodel_float = '{} -m {}  -w {} -d {} -q {} -e {} -t {} -o {} --input_config {} -p convert_cmodel_float'.format(
                                tool_path, model, w_file, minmax_path, chn_path, softmax_path, infect_datatype.file_path,
                                out_model, input_path)
            PopenRun(calibration_cmodel_float, out_model)
        if not (os.path.exists(out_model)):
            raise RuntimeError('Run convert cmodel float model failed!\nUse command to debug: {}'.format(calibration_cmodel_float))
    else:
        compileconfig_path = misc.find_path(dev_path, 'Dev_tool_CompilerConfig.txt')
        if re.search(r'_cmodel_float\.sim$', model, re.I):
            out_model = model.replace('_cmodel_float.sim', '_fixed.sim')
        elif re.search(r'_float\.sim$', model, re.I):
            out_model = model.replace('_float.sim', '_fixed.sim')
        else:
            out_model = model.replace('.sim', '_fixed.sim')
        if output is not None:
            if os.path.isdir(output):
                out_model = os.path.join(output, os.path.basename(out_model))
            elif (os.path.isdir(os.path.dirname(output)) or (os.path.dirname(output) == '')):
                out_model = output
        if debug:
            calibration_IPU = 'gdb --args {} -m {} -o {} -t {} --input_config {} --per_do_quant_threshold {} -p convert_fixed'.format(
                tool_path, model, out_model, compileconfig_path, input_path, quantThreshold)
            print('\033[33m================Debug command================\033[0m\n' + calibration_IPU + '\n\033[33m=============================================\033[0m')
            os.system(calibration_IPU)
        else:
            calibration_IPU = '{} -m {} -o {} -t {} --input_config {} --per_do_quant_threshold {} -p convert_fixed'.format(
                tool_path, model, out_model, compileconfig_path, input_path, quantThreshold)
            PopenRun(calibration_IPU, out_model)
        if not os.path.exists(out_model):
            raise RuntimeError('Run Calibration Fixed failed!\nUse command to debug: {}'.format(calibration_IPU))
        print('\033[31mFixed model at: {}\033[0m'.format(out_model))
    return out_model


def check_multi_statistics_after_sys(num_subset, signal, image_list, label, model, category, tool_path, model_name, preprocess_func, log=False):
    statistics_folder = 'statistics'
    if signal == 'minmax':
        filename = 'tensor_min_max.txt'
    elif signal == 'qab':
        filename = 'tensor_qab.txt'
    else:
        filename = 'tensor_statistics.txt'
    run_once = False
    for subset_index in range(num_subset):
        sub_dir = os.path.join(statistics_folder, 'sub_{:05d}'.format(subset_index))
        if not os.path.exists(os.path.join(sub_dir, filename)):
            print('\033[33m[ERROR] Phase {} Error!\033[0m'.format(signal))
            run_once = True
            break
    if run_once:
        sgs_calibration_statistics(signal, None, [image_list[0]], label, model, category, tool_path, model_name, preprocess_func, log=log)

