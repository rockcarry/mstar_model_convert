# -*- coding: utf-8 -*-

import time
import sys
import os
import cv2
import shutil
import importlib

cv2.setNumThreads(1)
image_suffix = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm',
                'sr', 'ras', 'tiff', 'tif', 'hdr', 'pic']


class ShowProcess(object):
    def __init__(self, max_steps, max_arrow=50):
        self.max_steps = max_steps
        self.i = 0
        self.max_arrow = max_arrow
        self.start = time.time()
        self.eta = 0.0
        self.total_time = 0.0
        self.last_time = self.start

    def elapsed_time(self):
        self.last_time = time.time()
        return self.last_time - self.start

    def calc_eta(self):
        elapsed = self.elapsed_time()
        if self.i == 0 or elapsed < 0.001:
            return None
        rate = float(self.i) / elapsed
        self.eta = (float(self.max_steps) - float(self.i)) / rate

    def get_time(self, _time):
        if (_time < 86400):
            return time.strftime("%H:%M:%S", time.gmtime(_time))
        else:
            s = (str(int(_time // 3600)) + ':' +
                 time.strftime("%M:%S", time.gmtime(_time)))
            return s

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        self.calc_eta()
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        if num_arrow < 2:
            process_bar = '\r' + '[' + '>' * num_arrow + ' ' * num_line + ']' \
                          + '%.2f' % percent + '%' + ' | ETA: ' + self.get_time(self.eta)
        elif num_arrow < self.max_arrow:
            process_bar = '\r' + '[' + '=' * (num_arrow-1) + '>' + ' ' * num_line + ']' \
                          + '%.2f' % percent + '%' + ' | ETA: ' + self.get_time(self.eta)
        else:
            process_bar = '\r' + '[' + '=' * num_arrow + ' ' * num_line + ']'\
                          + '%.2f' % percent + '%' + ' | ETA: ' + self.get_time(self.eta)
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        self.close()

    def close(self):
        if self.i >= self.max_steps:
            self.total_time = self.elapsed_time()
            print('\nTotal time elapsed: ' + self.get_time(self.total_time))


class Fake_Label(object):
    def __init__(self, model_name):
        label_txt = 'fake_label_{}.txt'.format(model_name)
        self.label_name = os.path.abspath(label_txt)
        with open(self.label_name, 'w') as fd:
            fd.write(' \n \n \n \n \n \n')

    def __del__(self):
        if os.path.exists(self.label_name):
            os.remove(self.label_name)


class Infect_Config(object):
    def __init__(self, config_path):
        self.file_path = config_path

    def blank(self):
        with open(self.file_path, 'w') as fd:
            pass

    def datatype(self):
        with open(self.file_path, 'w') as fd:
            fd.write('fuse_elementwise_to_preceding_ops\n')
            fd.write('fuse_activation_functions\n')
            fd.write('process_st_mimmax\n')
            fd.write('infect_datatype\n')
            fd.write('infect_quant_domain\n')
            fd.write('calcu_qab_minmax\n')

    def __del__(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


def find_path(path, name):
    if path.split('/')[-1] == name:
        return path
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.abspath(os.path.join(root, name))
    raise FileNotFoundError('File `{}` not found in directory `{}`'.format(name, os.path.abspath(path)))


def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if os.path.basename(apath).split('.')[-1].lower() in image_suffix:
                result.append(apath)
    if len(result) == 0:
        raise FileNotFoundError('No images found in {}'.format(dirname))
    return result


class Move_Log(object):
    def __init__(self, clean=True):
        self.clean = clean

    def __del__(self):
        if self.clean:
            renew_folder('log')
            txts = ['tensor_min_max.txt', 'tensor_qab.txt', 'tensor_statistics.txt', 'tensor_statistics_type.txt',
                    'softmax_lut_range.txt', 'tensor_weight.txt', 'tensor_weight_calibration.txt', 'tensor_chn.txt']
            logs = ['statistics.log', 'simulator.log', 'offline.log', 'float.log']
            if os.path.exists('output'):
                shutil.move('output', 'log')
            if os.path.exists('statistics'):
                shutil.move('statistics', 'log')
            if os.path.exists('tmp_image'):
                shutil.move('tmp_image', 'log')
            if os.path.exists('statistics'):
                shutil.move('statistics', 'log')
            file_list = os.listdir(os.getcwd())
            for item in file_list:
                if item.split('/')[-1] in txts:
                    shutil.move(item, 'log')
                if item.split('_')[-1] in logs:
                    shutil.move(item, 'log')


def image_preprocess_func(model_name):
    if 'SGS_IPU_DIR' in os.environ:
        project_path = os.environ['SGS_IPU_DIR']
        preprocess_path = os.path.join(project_path, 'Scripts/calibrator/preprocess_method')
    elif 'TOP_DIR' in os.environ:
        project_path = os.environ['TOP_DIR']
        preprocess_path = os.path.join(project_path, 'SRC/Tool/Scripts/calibrator/preprocess_method')
    else:
        raise OSError('Run `source cfg_env.sh` in top directory.')

    if os.path.exists(model_name) and model_name.split('.')[-1] == 'py':
        sys.path.append(os.path.dirname(model_name))
        preprocess_func = importlib.import_module(os.path.basename(model_name).split('.')[0])
    else:
        sys.path.append(os.path.abspath(preprocess_path))
        preprocess_func = importlib.import_module(model_name)
    return preprocess_func.image_preprocess


def convert_image(img_path, output_img, preprocess_func, norm=True):
    image = preprocess_func(img_path, norm)
    img = list(image.flat)
    img_origin = cv2.imread(img_path, flags=-1)
    if img_origin is None:
        hw_str = '{}, {}'.format(image.shape[0], image.shape[0])
    else:
        hw_str = '{}, {}'.format(img_origin.shape[0], img_origin.shape[1])
    with open(output_img, 'w') as f:
        f.write('Input_shape=%s\n' % hw_str)
        for num, value in enumerate(img):
            f.write('{}, '.format(value))
            if (num + 1) % 16 == 0:
                f.write('\n')


def check_model_name(img_path, model_name):
    img = eval(model_name).image_preprocess(img_path)


def renew_folder(folder_name):
    if os.path.exists(folder_name):
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        else:
            os.remove(folder_name)
            os.mkdir(folder_name)
    else:
        os.mkdir(folder_name)


def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    if len(result) == 0:
        raise FileNotFoundError()
    return result


