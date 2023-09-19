# -*- coding: utf-8 -*-

import time
import sys
import os
import re
import cv2
import yaml
import shutil
import importlib
import subprocess
import numpy as np
from multiprocessing import Process
from calibrator_custom.versions import VERSION

cv2.setNumThreads(1)
os.environ['OPENBLAS_NUM_THREADS'] = '1'

image_suffix = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm',
                'sr', 'ras', 'tiff', 'tif', 'hdr', 'pic', 'raw', 'npy', 'bin', 'mp4']

x64_extension_suffix = '.cpython-37m-x86_64-linux-gnu.so'
i386_extension_suffix = '.cpython-37m-i386-linux-gnu.so'


def get_sdk_version():
    return VERSION.split('.')[0]


def image_preprocess_func(model_name):
    if 'SGS_IPU_DIR' in os.environ:
        project_path = os.environ['SGS_IPU_DIR']
        preprocess_path = os.path.join(project_path, 'Scripts/calibrator/preprocess_method')
    elif 'IPU_TOOL' in os.environ:
        project_path = os.environ['IPU_TOOL']
        preprocess_path = os.path.join(project_path, 'Scripts/calibrator/preprocess_method')
    else:
        raise OSError('Run `source cfg_env.sh` in top directory.')

    if model_name.split('.')[-1] == 'py':
        if not os.path.exists(model_name):
            raise FileNotFoundError('Can not find preprocess file: {}'.format(model_name))
        sys.path.append(os.path.dirname(model_name))
        preprocess_func = importlib.import_module(os.path.basename(model_name).split('.')[0])
    else:
        sys.path.append(os.path.abspath(preprocess_path))
        preprocess_func = importlib.import_module(model_name)
    return preprocess_func.image_preprocess


def image_generator(image_list, preprocess_funcs, norm=True):
    for image in image_list:
        imgs = []
        if len(preprocess_funcs) > 1:
            if len(preprocess_funcs) != len(image):
                raise ValueError('Got different num of preprocess_methods and images!')
            for idx, preprocess_func in enumerate(preprocess_funcs):
                imgs.append(preprocess_func(image[idx], norm))
            yield [imgs]
        else:
            if isinstance(image, list):
                if len(preprocess_funcs) != 1 or len(image) != 1:
                    raise ValueError('Got different num of preprocess_methods and images!')
                imgs.append(preprocess_funcs[0](image[0], norm))
                yield [imgs]
            else:
                imgs.append(preprocess_funcs[0](image, norm))
                yield [imgs]


class CompilerConfig(object):
    def __init__(self, useFile=False):
        self.FilePath = []
        self.useFile = useFile
        if 'SGS_IPU_DIR' in os.environ:
            project_path = os.environ['SGS_IPU_DIR']
            if get_sdk_version() in ['1', 'Q_0']:
                self.CompilerConfigPath = os.path.join(project_path, 'cfg/CompilerConfig.yaml')
            else:
                self.CompilerConfigPath = os.path.join(project_path, 'cfg/CompilerConfigS.yaml')
        elif 'IPU_TOOL' in os.environ:
            project_path = os.environ['IPU_TOOL']
            if get_sdk_version() in ['1', 'Q_0']:
                self.CompilerConfigPath = os.path.join(project_path, 'cfg/CompilerConfig.yaml')
            else:
                self.CompilerConfigPath = os.path.join(project_path, 'cfg/CompilerConfigS.yaml')
        else:
            raise OSError('Run `source cfg_env.sh` in top directory.')
        with open(self.CompilerConfigPath, 'r') as f:
            self.CompilerPass = yaml.load(f, Loader=yaml.Loader)

    def Debug2FloatConfig(self, SkipInfectDtype=False):
        FloatPassPath = 'Debug2FloatConfig.txt'
        FloatPass = self.CompilerPass['DEBUG_2_FLOAT_CONFIG']
        if SkipInfectDtype:
            FloatPass.remove('infect_datatype')
        if self.useFile:
            self.FilePath.append(FloatPassPath)
            with open(FloatPassPath, 'w') as f:
                f.writelines([i + '\n' for i in FloatPass])
            return FloatPassPath
        return FloatPass

    def Float2CmodelFloatConfig(self):
        CmodelFloatPassPath = 'Float2CmodelFloatConfig.txt'
        CmodelFloatPass = self.CompilerPass['FLOAT_2_CMODEL_FLOAT_CONFIG']
        if self.useFile:
            self.FilePath.append(CmodelFloatPassPath)
            with open(CmodelFloatPassPath, 'w') as f:
                f.writelines([i + '\n' for i in CmodelFloatPass])
            return CmodelFloatPassPath
        return CmodelFloatPass

    def CmodelFloat2FixedConfig(self):
        FixedPassPath = 'CmodelFloat2FixedConfig.txt'
        FixedPass = self.CompilerPass['CMODEL_FLOAT_2_FIXED_CONFIG']
        if self.useFile:
            self.FilePath.append(FixedPassPath)
            with open(FixedPassPath, 'w') as f:
                f.writelines([i + '\n' for i in FixedPass])
            return FixedPassPath
        return FixedPass

    def CmodelFloat2FixedWOConfig(self):
        CmodelFloat2FixedWOPassPath = 'CmodelFloat2FixedWOConfig.txt'
        CmodelFloat2FixedWOPass = self.CompilerPass['CMODEL_FLOAT_2_FIXEDWO_CONFIG']
        if self.useFile:
            self.FilePath.append(CmodelFloat2FixedWOPassPath)
            with open(CmodelFloat2FixedWOPassPath, 'w') as f:
                f.writelines([i + '\n' for i in CmodelFloat2FixedWOPass])
            return CmodelFloat2FixedWOPassPath
        return CmodelFloat2FixedWOPass

    def FixedWO2Fixed(self):
        FixedWO2FixedPassPath = 'FixedWO2FixedConfig.txt'
        FixedWO2FixedPass = self.CompilerPass['FIXEDWO_2_FIXED_CONFIG']
        if self.useFile:
            self.FilePath.append(FixedWO2FixedPassPath)
            with open(FixedWO2FixedPassPath, 'w') as f:
                f.writelines([i + '\n' for i in FixedWO2FixedPass])
            return FixedWO2FixedPassPath
        return FixedWO2FixedPass

    def Float2CmodelFloatVerifyConfig(self):
        CmodelFloatVerifyPassPath = 'Float2CmodelFloatVerifyConfig.txt'
        CmodelFloatVerifyPass = self.CompilerPass['FLOAT_2_CMODEL_FLOAT_VERIFY_CONFIG']
        if self.useFile:
            self.FilePath.append(CmodelFloatVerifyPassPath)
            with open(CmodelFloatVerifyPassPath, 'w') as f:
                f.writelines([i + '\n' for i in CmodelFloatVerifyPass])
            return CmodelFloatVerifyPassPath
        return CmodelFloatVerifyPass

    def MixedOptimConfig(self):
        OptimPassPath = 'MixedOptimConfig.txt'
        OptimPass = self.CompilerPass['8_16_MIX_OPTIM_CONFIG']
        if self.useFile:
            self.FilePath.append(OptimPassPath)
            with open(OptimPassPath, 'w') as f:
                f.writelines([i + '\n' for i in OptimPass])
            return OptimPassPath
        return OptimPass

    def __del__(self):
        for file in self.FilePath:
            if os.path.exists(file):
                os.remove(file)


class ShowProcess(object):
    def __init__(self, max_steps, max_arrow=50):
        self.max_steps = max_steps
        self.i = 0
        self.max_arrow = max_arrow
        self.start = time.time()
        self.eta = 0.0
        self.total_time = 0.0
        self.last_time = self.start
        self.convert_process = None
        if self.max_steps == 0:
            print('\033[33mThe input data is empty, please check the input parameters!\033[0m')
            raise ValueError('The input data is empty, please check the input parameters!')

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

    def show_converting(self, ppid):
        converting_info = ['\rStart converting model', '\rStart converting model.',
                           '\rStart converting model..', '\rStart converting model...']
        count = 0
        while os.getppid() == ppid:
            sys.stdout.write(converting_info[count])
            sys.stdout.flush()
            time.sleep(0.2)
            count += 1
            if count >= 4:
                count = 0

    def start_show_convert(self):
        self.convert_process = Process(target=self.show_converting, args=(os.getpid(),), daemon=True)
        self.convert_process.start()

    def close_show_convert(self):
        if self.convert_process is not None and self.convert_process.is_alive():
            self.convert_process.terminate()


def check_quant_param(quant_param):
    if quant_param is not None:
        if not isinstance(quant_param, list):
            raise ValueError('quant_param must be a `list`')
        for idx, item in enumerate(quant_param):
            if not isinstance(item['name'], str):
                raise ValueError('name of each quant_param[{}] must be `str`'.format(idx))
            if not isinstance(item['min'], list):
                raise ValueError('min of quant_param[{}]: {} must be a `list`'.format(idx, item['name']))
            if not isinstance(item['max'], list):
                raise ValueError('max of quant_param[{}]: {} must be a `list`'.format(idx, item['name']))
            if not isinstance(item['bit'], int):
                raise ValueError('bit of quant_param[{}]: {} must be `int`'.format(idx, item['name']))
            if 'data' in item:
                if not isinstance(item['data'], np.ndarray):
                    raise ValueError('data of quant_param[{}]: {} must be `numpy.ndarray`'.format(idx, item['name']))


def get_out_model_name(model_path, output=None, phase='Fixed'):
    if phase == 'Fixed':
        if re.search(r'((_cmodel|)_float|_fixed_without_ipu_ctrl)\.sim$', model_path, re.I):
            out_model = re.sub(r'((_cmodel|)_float|_fixed_without_ipu_ctrl)\.sim$', '_fixed.sim', model_path)
        else:
            out_model = model_path.replace('.sim', '_fixed.sim')
        if output is not None:
            if os.path.isdir(output):
                out_model = os.path.join(output, os.path.basename(out_model))
            elif (os.path.isdir(os.path.dirname(output)) or (os.path.dirname(output) == '')):
                out_model = output
    elif phase == 'Fixed_without_ipu_ctrl':
        if re.search(r'(_cmodel|)_float\.sim$', model_path, re.I):
            out_model = re.sub(r'(_cmodel|)_float\.sim$', '_fixed_without_ipu_ctrl.sim', model_path)
        else:
            out_model = model_path.replace('.sim', '_fixed_without_ipu_ctrl.sim')
        if output is not None:
            if os.path.isdir(output):
                out_model = os.path.join(output, os.path.basename(out_model))
            elif (os.path.isdir(os.path.dirname(output)) or (os.path.dirname(output) == '')):
                out_model = output
    else:
        if re.search(r'_float\.sim$', model_path, re.I):
            out_model = model_path.replace('_float.sim', '_cmodel_float.sim')
        else:
            out_model = model_path.replace('.sim', '_cmodel_float.sim')
        if output is not None:
            if os.path.isdir(output):
                out_model = os.path.join(output, os.path.basename(out_model))
            elif (os.path.isdir(os.path.dirname(output)) or (os.path.dirname(output) == '')):
                out_model = output
    return out_model


def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if os.path.basename(apath).split('.')[-1].lower() in image_suffix:
                result.append(os.path.abspath(apath))
    if len(result) == 0:
        raise FileNotFoundError('No images found in {}'.format(dirname))
    return result


def _to_yuv420(image, input_details):
    def _to_double(image):
        if image.shape[0] % 2 != 0:
            img_1 = np.zeros((image.shape[0] + 1, image.shape[1]))
            img_1[:-1, :] = image
            img_1[-1, :] = image[-1, :]
        else:
            img_1 = image
        if img_1.shape[1] % 2 != 0:
            img_2 = np.zeros((img_1.shape[0], img_1.shape[1] + 1))
            img_2[:, :-1] = img_1
            img_2[:, -1] = img_1[:, -1]
        else:
            img_2 = img_1
        return img_2

    align_up = lambda x, align: (x // align + 1) * align if (x % align != 0) else (x // align) * align
    rgb2yuv_covert_matrix = np.array(
        [[218.0/1024, 732.0/1024, 74.0/1024],
         [-117.0/1024, -395.0/1024, 512.0/1024],
         [512.0/1024, -465.0/1024, -47.0/1024]])
    if input_details['training_input_formats'] == 'BGR':
        if input_details['layouts'] == 'NCHW':
            image = np.transpose(image, axes=(1, 2, 0))
        img_rgb = image[:, :, [2, 1, 0]]
    elif input_details['training_input_formats'] == 'RGB':
        if input_details['layouts'] == 'NCHW':
            image = np.transpose(image, axes=(1, 2, 0))
        img_rgb = image
    elif input_details['training_input_formats'] == 'GRAY':
        if input_details['layouts'] == 'NCHW':
            image = np.transpose(image, axes=(1, 2, 0))
        h = align_up(image.shape[0], input_details['input_height_alignment'])
        w = align_up(image.shape[1], input_details['input_width_alignment'])
        img_align = np.zeros((h, w, 1))
        img_align[:image.shape[0], :image.shape[1], :] = image
        out_img = np.zeros(input_details['shape'][1:]).flatten()
        out_img[:img_align.size] = img_align.flatten()
        return out_img.reshape(input_details['shape'][1:])
    else:
        raise ValueError('Not support this training_input_formats: {}'.format(
            input_details['training_input_formats']))
    yuv444_y = np.dot(img_rgb, rgb2yuv_covert_matrix[0, :])
    yuv444_u = np.add(np.dot(img_rgb, rgb2yuv_covert_matrix[1, :]), 128)
    yuv444_v = np.add(np.dot(img_rgb, rgb2yuv_covert_matrix[2, :]), 128)

    yuv444_y_2 = _to_double(yuv444_y)
    yuv444_u_2 = _to_double(yuv444_u)
    yuv444_v_2 = _to_double(yuv444_v)

    yuv420_u = np.add(yuv444_u_2[::2, :], yuv444_u_2[1::2, :])
    yuv420_u = np.add(yuv420_u[:, ::2], yuv420_u[:, 1::2])
    yuv420_u = np.divide(yuv420_u, 4)
    yuv420_v = np.add(yuv444_v_2[::2, :], yuv444_v_2[1::2, :])
    yuv420_v = np.add(yuv420_v[:, ::2], yuv420_v[:, 1::2])
    yuv420_v = np.divide(yuv420_v, 4)
    yuv420_uv = np.zeros((yuv420_u.shape[0], yuv420_u.shape[1] * 2))
    yuv420_uv[:, 0::2] = yuv420_u
    yuv420_uv[:, 1::2] = yuv420_v
    yuv420 = np.clip(np.floor(np.vstack([yuv444_y_2, yuv420_uv]) + 0.5), 0, 255)
    return np.expand_dims(yuv420, -1)


def convert_to_input_formats(image, input_details):
    if 'training_input_formats' not in input_details or 'input_formats' not in input_details:
        return image
    if input_details['input_formats'] != 'RAWDATA_S16_NHWC' and image.dtype != input_details['dtype']:
        raise ValueError('Got tensor of type {} but expected type {} for input: {}'.format(
            image.dtype, str(input_details['dtype']), input_details['name']))
    if len(image.shape) != len(input_details['shape']):
        raise ValueError('Dimension length mismatch. Got {} but expected {} for input: {}'.format(
            len(image.shape), len(input_details['shape']), input_details['name']))
    if input_details['input_formats'] == 'YUV_NV12':
        img_yuv = []
        for idx in range(image.shape[0]):
            img = image[idx]
            img_yuv.append(np.expand_dims(_to_yuv420(img, input_details), 0))
        np_img_yuv = np.concatenate(img_yuv)
        out_img = np.zeros(input_details['shape'])
        out_img[:, :np_img_yuv.shape[1], :np_img_yuv.shape[2], :] = np_img_yuv
        return out_img.astype(input_details['dtype'])
    elif input_details['input_formats'] == 'GRAY':
        if input_details['training_input_formats'] != 'RGB' and \
            input_details['training_input_formats'] != 'GRAY':
            raise ValueError('GRAY input_formats only support RGB/GRAY for training_input_formats!')
        gray = np.zeros(input_details['shape'])
        if input_details['layouts'] == 'NCHW':
            gray[:, :, :image.shape[2], :image.shape[3]] = image
        else:
            gray[:, :image.shape[1], :image.shape[2], :] = image
        return gray.astype(input_details['dtype'])
    elif input_details['input_formats'] == 'RGBA':
        img_rgbas = []
        for idx in range(image.shape[0]):
            img = image[idx]
            if input_details['training_input_formats'] == 'BGR':
                img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            elif input_details['training_input_formats'] == 'RGB':
                img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            elif input_details['training_input_formats'] == 'RGBA':
                img_rgba = img
            else:
                raise ValueError('Not support this training_input_formats: {}'.format(
                    input_details['training_input_formats']))
            img_rgbas.append(np.expand_dims(img_rgba, 0))
        np_img_rgbas = np.concatenate(img_rgbas)
        out_img = np.zeros(input_details['shape'])
        out_img[:, :np_img_rgbas.shape[1], :np_img_rgbas.shape[2], :] = np_img_rgbas
        return out_img.astype(input_details['dtype'])
    elif input_details['input_formats'] == 'BGRA':
        img_bgras = []
        for idx in range(image.shape[0]):
            img = image[idx]
            if input_details['training_input_formats'] == 'BGR':
                img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            elif input_details['training_input_formats'] == 'RGB':
                img_bgra = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
            elif input_details['training_input_formats'] == 'BGRA':
                img_bgra = img
            else:
                raise ValueError('Not support this training_input_formats: {}'.format(
                    input_details['training_input_formats']))
            img_bgras.append(np.expand_dims(img_bgra, 0))
        np_img_bgras = np.concatenate(img_bgras)
        out_img = np.zeros(input_details['shape'])
        out_img[:, :np_img_bgras.shape[1], :np_img_bgras.shape[2], :] = np_img_bgras
        return out_img.astype(input_details['dtype'])
    elif input_details['input_formats'] == 'RAWDATA_S16_NHWC':
        if input_details['training_input_formats'] != 'RAWDATA_S16_NHWC':
            raise ValueError('RAWDATA_S16_NHWC input_formats only support RAWDATA_S16_NHWC for training_input_formats!')
        if image.dtype != np.float32:
            raise ValueError('RAWDATA_S16_NHWC input_formats need float32 datatype for convert!')
        scale, _ = input_details['quantization']
        feature = np.clip((image / scale), -32767, 32767).astype(input_details['dtype'])
        feature_s16 = np.zeros(input_details['shape']).astype(input_details['dtype'])
        feature_s16[..., :feature.shape[-1]] = feature
        return feature_s16
    elif input_details['input_formats'] == 'BGR':
        if input_details['training_input_formats'] != 'BGR':
            raise ValueError('BGR input_formats only support BGR for training_input_formats!')
        return image
    elif input_details['input_formats'] == 'RGB':
        if input_details['training_input_formats'] != 'RGB':
            raise ValueError('RGB input_formats only support RGB for training_input_formats!')
        return image
    else:
        raise ValueError('Not support this input_formats: {}'.format(input_details['input_formats']))


