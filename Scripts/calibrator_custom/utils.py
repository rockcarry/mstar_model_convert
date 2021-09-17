# -*- coding: utf-8 -*-

import time
import sys
import os
import re
import cv2
import shutil
import importlib
import numpy as np

cv2.setNumThreads(1)
image_suffix = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm',
                'sr', 'ras', 'tiff', 'tif', 'hdr', 'pic']

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


def get_compiler_config():
    if 'SGS_IPU_DIR' in os.environ:
        project_path = os.environ['SGS_IPU_DIR']
        compiler_config_path = os.path.join(project_path, 'cfg/CompilerConfig.txt')
    elif 'TOP_DIR' in os.environ:
        project_path = os.environ['TOP_DIR']
        compiler_config_path = os.path.join(project_path, 'SRC/Tool/cfg/CompilerConfig.txt')
    else:
        raise OSError('Run `source cfg_env.sh` in top directory.')
    return compiler_config_path


def get_new_compiler_config():
    if 'SGS_IPU_DIR' in os.environ:
        project_path = os.environ['SGS_IPU_DIR']
        compiler_config_path = os.path.join(project_path, 'cfg/CompilerConfig.txt')
    elif 'TOP_DIR' in os.environ:
        project_path = os.environ['TOP_DIR']
        compiler_config_path = os.path.join(project_path, 'SRC/Tool/cfg/CompilerConfig.txt')
    else:
        raise OSError('Run `source cfg_env.sh` in top directory.')
    with open(compiler_config_path, 'r') as f:
        compiler_config = f.readlines()
    compiler_config.remove('infect_datatype\n')
    new_compiler_config_path = os.path.basename(compiler_config_path).replace('CompilerConfig', 'NewCompilerConfig')
    with open(new_compiler_config_path, 'w') as f:
        f.writelines(compiler_config)
    return new_compiler_config_path


def get_dev_config():
    if 'SGS_IPU_DIR' in os.environ:
        project_path = os.environ['SGS_IPU_DIR']
        dev_config_path = os.path.join(project_path, 'cfg/Dev_tool_CompilerConfig.txt')
    elif 'TOP_DIR' in os.environ:
        project_path = os.environ['TOP_DIR']
        dev_config_path = os.path.join(project_path, 'SRC/Tool/cfg/Dev_tool_CompilerConfig.txt')
    else:
        raise OSError('Run `source cfg_env.sh` in top directory.')
    return dev_config_path


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
        if re.search(r'_float\.sim$', model_path, re.I):
            out_model = model_path.replace('_float.sim', '_fixed.sim')
        else:
            out_model = model_path.replace('.sim', '_fixed.sim')
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
                result.append(apath)
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

    rgb2yuv_covert_matrix = np.array(
        [[218.0/1024, 732.0/1024, 74.0/1024],
         [-117.0/1024, -395.0/1024, 512.0/1024],
         [512.0/1024, -465.0/1024, -47.0/1024]])
    img = np.squeeze(image, 0)
    if input_details['training_input_formats'] == 'BGR':
        img_rgb = img[:, :, [2, 1, 0]]
    elif input_details['training_input_formats'] == 'RGB':
        img_rgb = img
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
    yuv420 = np.clip(np.round(np.vstack([yuv444_y_2, yuv420_uv])), 0, 255)
    out_yuv = np.zeros(input_details['shape'])
    out_yuv[0, :yuv420.shape[0], :yuv420.shape[1], 0] = yuv420
    return out_yuv.astype(input_details['dtype'])


def convert_to_input_formats(image, input_details):
    if 'training_input_formats' not in input_details or 'input_formats' not in input_details:
        return image
    if input_details['input_formats'] != 'RAWDATA_S16_NHWC' and image.dtype != input_details['dtype']:
        raise ValueError('Got tensor of type {} but expected type {} for input: {}'.format(
            image.dtype, input_details['dtype'], input_details['name']))
    if len(image.shape) != len(input_details['shape']):
        raise ValueError('Dimension length mismatch. Got {} but expected {} for input: {}'.format(
            len(image.shape), len(input_details['shape']), input_details['name']))
    if input_details['input_formats'] == 'YUV_NV12':
        return _to_yuv420(image, input_details)
    elif input_details['input_formats'] == 'GRAY':
        if input_details['training_input_formats'] != 'RGB':
            raise ValueError('GRAY input_formats only support RGB for training_input_formats!')
        gray = np.zeros(input_details['shape'])
        gray[:, :image.shape[1], :image.shape[2], :] = image
        return gray.astype(input_details['dtype'])
    elif input_details['input_formats'] == 'RGBA':
        img = np.squeeze(image, 0)
        if input_details['training_input_formats'] == 'BGR':
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif input_details['training_input_formats'] == 'RGB':
            img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        else:
            raise ValueError('Not support this training_input_formats: {}'.format(
                input_details['training_input_formats']))
        img = np.zeros(input_details['shape'])
        img[0, :img_rgba.shape[0], :img_rgba.shape[1], :] = img_rgba
        return img.astype(input_details['dtype'])
    elif input_details['input_formats'] == 'BGRA':
        img = np.squeeze(image, 0)
        if input_details['training_input_formats'] == 'BGR':
            img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        elif input_details['training_input_formats'] == 'RGB':
            img_bgra = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
        else:
            raise ValueError('Not support this training_input_formats: {}'.format(
                input_details['training_input_formats']))
        img = np.zeros(input_details['shape'])
        img[0, :img_bgra.shape[0], :img_bgra.shape[1], :] = img_bgra
        return img.astype(input_details['dtype'])
    elif input_details['input_formats'] == 'RAWDATA_S16_NHWC':
        if input_details['training_input_formats'] != 'RAWDATA_S16_NHWC':
            raise ValueError('RAWDATA_S16_NHWC input_formats only support RAWDATA_S16_NHWC for training_input_formats!')
        if image.dtype != np.float32:
            raise ValueError('RAWDATA_S16_NHWC input_formats need float32 datatype for convert!')
        scale, _ = input_details['quantization']
        feature = np.clip((image / scale), -32768, 32767).astype(input_details['dtype'])
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


