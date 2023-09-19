# -*- coding: utf-8 -*-

import os
import functools
import numpy as np
import calibrator_custom
from calibrator_custom import utils
import configparser
import string

ALL_PLATFORM = ['1', 'Q_0', 'S6', 'S2', 'S1', 'S3', 'S31', 'L10', 'S02']
FP32_OPERATOR = ['S1']
IPU2_0_PLATFORM = ['S6', 'S2', 'S1', 'S3', 'S31', 'L10', 'S02']
LAYERS_COMMON = {
    'Add': ALL_PLATFORM,
    'Sub': ALL_PLATFORM,
    'Mul': ALL_PLATFORM,
    'Input': ALL_PLATFORM,
    'Reshape': ALL_PLATFORM,
    'Greater': ALL_PLATFORM,
    'Less': ALL_PLATFORM,
    'Select': ALL_PLATFORM,
    'Conv2d': ALL_PLATFORM,
    'Average_Pool_2D': ALL_PLATFORM,
    'Max_Pool_2D': ALL_PLATFORM,
    'DepthWiseConv2D': ALL_PLATFORM,
    'BatchMatMul': ALL_PLATFORM,
    'Pad': ALL_PLATFORM,
    'Softmax': ALL_PLATFORM,
    'Logistic': ALL_PLATFORM,
    'Tanh': ALL_PLATFORM,
    'Relu': ALL_PLATFORM,
    'Relu6': ALL_PLATFORM,
    'LeakyRelu': ALL_PLATFORM,
    'Relu_N1_TO_1': ALL_PLATFORM,
    'Prelu': ALL_PLATFORM,
    'Mean': ALL_PLATFORM,
    'Sum': ALL_PLATFORM,
    'Reduce_Max': ALL_PLATFORM,
    'TopKV2': ALL_PLATFORM,
    'Round': ALL_PLATFORM,
    'Sqrt': ALL_PLATFORM,
    'Fullyconnected': ALL_PLATFORM,
    'Abs': ALL_PLATFORM,
    'Reciprocal': ALL_PLATFORM,
    'Elu': ALL_PLATFORM,
    'Equal': ALL_PLATFORM,
    'NotEqual': ALL_PLATFORM,
    'GreaterEqual': ALL_PLATFORM,
    'LogicalAnd': ALL_PLATFORM,
    'LogicalNot': ALL_PLATFORM,
    'Maximum': ALL_PLATFORM,
    'Minimum': ALL_PLATFORM,
    'Exp': ALL_PLATFORM,
    'Slice': ALL_PLATFORM,
    'StridedSlice': ALL_PLATFORM,
    'Unpack': ALL_PLATFORM,
    'Pack': ALL_PLATFORM,
    'Tile': ALL_PLATFORM,
    'Transpose': ALL_PLATFORM,
    'Split': ALL_PLATFORM,
    'Split_V': ALL_PLATFORM,
    'Gather': ALL_PLATFORM,
    'Concatenation': ALL_PLATFORM,
    'TFLite_Detection_NMS': ALL_PLATFORM,
    'PostProcess_Unpack': ALL_PLATFORM,
    'PostProcess_Max': ALL_PLATFORM,
    'BoxDecoder': ALL_PLATFORM,
    'BoxDecoder2': ALL_PLATFORM,
    'Fix2Float': ['Q_0', 'S6', 'S2', 'S1', 'S3', 'S31', 'L10', 'S02'],
    'Float2Fix': FP32_OPERATOR,
    'FFT': FP32_OPERATOR,
    'IFFT': FP32_OPERATOR,
    'RFFT': FP32_OPERATOR,
    'IRFFT': FP32_OPERATOR,
    'FFT2': FP32_OPERATOR,
    'IFFT2': FP32_OPERATOR,
    'FLOAT2COMPLEX': FP32_OPERATOR,
    'ExtractReal': FP32_OPERATOR,
    'AbsFp32': FP32_OPERATOR,
    'SubFp32': FP32_OPERATOR,
    'MulFp32': FP32_OPERATOR,
    'DivFp32': FP32_OPERATOR,
    'AddFp32': FP32_OPERATOR,
    'SumFp32': FP32_OPERATOR,
    'MulCN': FP32_OPERATOR,
    'ConjCN': FP32_OPERATOR,
    'ConjMulCN': FP32_OPERATOR,
    'AddCN': FP32_OPERATOR,
    'SubCN': FP32_OPERATOR,
    'DivCN': FP32_OPERATOR,
    'Matmul': FP32_OPERATOR,
    'MedianFilter': FP32_OPERATOR,
    'ConvFilter': FP32_OPERATOR,
    'TraceMatrix': FP32_OPERATOR,
    'CustomizedScatterND': IPU2_0_PLATFORM,
    'Softplus': IPU2_0_PLATFORM,
    'Square': ALL_PLATFORM,
    'Atan2': ALL_PLATFORM,
    'ReduceMin': ALL_PLATFORM,
    'CustomPow': IPU2_0_PLATFORM,
    'Div': IPU2_0_PLATFORM,
    'Sin': ALL_PLATFORM,
    'Cos': ALL_PLATFORM,
    'DmaCoefMatrix': ALL_PLATFORM,
    'FloorDiv': ['1', 'Q_0'],
}

def mace_check(condition, msg):
    if not condition:
        raise Exception(msg)


def platform_register(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        sdk_version = calibrator_custom.utils.get_sdk_version()
        if sdk_version not in LAYERS_COMMON[func.__name__]:
            raise ReferenceError('Operator [{}] not support in this version: {}'.format(func.__name__, calibrator_custom.__version__))
        return func(*args, **kw)
    return wrapper


def check_elementwise_tensor_shape(x_shape, y):
    if len(x_shape) != len(y.shape):
        if not (len(y.shape) == 2 and y.shape[0] == 1 and y.shape[1] == 1):
            if len(y.shape) != 1:
                raise ValueError('ElementWise only support inner most dimension broadcasting.')
            if not (y.shape[-1] == 1 or y.shape[-1] == x_shape[-1]):
                raise ValueError('ElementWise broadcasting only support inner most dimension is 1 or x inner most dimension.')
    else:
        for idx, xi in enumerate(x_shape):
            if xi != y.shape[idx]:
                raise ValueError('ElementWise not support such Tensors: {} - {}'.format(x_shape, y.shape))


def convert_dtype_to_bit(dtype):
    if dtype == 'uint8':
        return 8
    elif dtype == 'int16':
        return 16
    elif dtype == 'float32':
        return 33
    elif dtype == 'int32':
        return 32
    elif dtype == 'int64':
        return 64
    elif dtype == 'complex64':
        return 65
    else:
        raise ValueError('Not support data type:', dtype)


class Chalk_Calibrator(calibrator_custom.SIM_Calibrator):
    def __init__(self, model_path, input_config, log=False):
        super().__init__()
        self.model = calibrator_custom.calibrator(model_path, input_config, show_log=log)

    def forward(self, x):
        in_details = self.model.get_input_details()
        out_details = self.model.get_output_details()
        for idx, _ in enumerate(in_details):
            self.model.set_input(idx, x[idx])
        self.model.invoke()
        result_list = []
        for idx, _ in enumerate(out_details):
            result = self.model.get_output(idx)
            result_list.append(result)
        return result_list


def print_model(model):
    main_str = 'Chalk_Calibrator:\n'
    main_str += 'model ' + calibrator_custom.SIM_Calibrator.print_model(model)
    print(main_str)


def get_chalk_input_tensor(name, chalk_model):
    for idx, chalk_tensor in enumerate(chalk_model.inputs):
        if name == chalk_tensor.name:
            return chalk_tensor
    return None


def autogen_preprocess(model, input_config):
    sdk_version = calibrator_custom.utils.get_sdk_version()
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(input_config, encoding='utf-8')
    input_arrays_str = config['INPUT_CONFIG']['inputs'].strip(string.punctuation)
    input_arrays = [s.strip() for s in input_arrays_str.split(',')]
    output_arrays_str = config['OUTPUT_CONFIG']['outputs'].strip(string.punctuation)
    output_arrays = [s.strip() for s in output_arrays_str.split(',')]
    input_formats_str = config['INPUT_CONFIG']['input_formats'].strip(string.punctuation)
    input_formats = [s.strip() for s in input_formats_str.split(',')]
    training_input_formats_str = config['INPUT_CONFIG'].get('training_input_formats', input_formats_str)
    training_input_formats = [s.strip() for s in training_input_formats_str.strip(string.punctuation).split(',')]
    model_inputs = model.get_input_details()
    model_outputs = model.get_output_details()

    if len(input_arrays) != len(input_formats) or len(input_arrays) != len(training_input_formats):
        raise ValueError('{} not right: Got {} input_arrays, {} input_formats, {} training_input_formats.'.format(
            input_config, len(input_arrays), len(input_formats), len(training_input_formats)))

    if len(input_arrays) != len(model_inputs) or len(output_arrays) != len(model_outputs):
        raise ValueError('input_config.ini is not compatible with model!')

    if 'RGB' in training_input_formats or 'BGR' in training_input_formats or 'GRAY' in training_input_formats:
        mean_str = config['INPUT_CONFIG'].get('mean', None)
        if mean_str is not None:
            mean = [[float(i) for i in s.split(':')] for s in mean_str.strip(string.punctuation).split(',')]
        else:
            mean_red_str = config['INPUT_CONFIG'].get('mean_red', None)
            mean_green_str = config['INPUT_CONFIG'].get('mean_green', None)
            mean_blue_str = config['INPUT_CONFIG'].get('mean_blue', None)
            if mean_red_str is None or mean_green_str is None or mean_blue_str is None:
                raise ValueError('training_input_formats is {} need set mean/std value.'.format(training_input_formats))
            mean_red = [float(i) for i in mean_red_str.strip(string.punctuation).split(',')]
            mean_green = [float(i) for i in mean_green_str.strip(string.punctuation).split(',')]
            mean_blue = [float(i) for i in mean_blue_str.strip(string.punctuation).split(',')]
            mean = [[mean_red[i], mean_green[i], mean_blue[i]] for i, _ in enumerate(mean_red)]
        std_str = config['INPUT_CONFIG']['std_value']
        std = [[float(i) for i in s.split(':')] for s in std_str.strip(string.punctuation).split(',')]

    preprocess_scripts = []
    for idx, _ in enumerate(input_arrays):
        model_preprocess = dict()
        model_preprocess['training_input_formats'] = training_input_formats[idx]
        if input_arrays[idx] == model_inputs[idx]['name']:
            file_name = 'autogen_input{}_preprocess.py'.format(idx)
            with open(file_name, 'w') as fw:
                if training_input_formats[idx] in ['BGR', 'RGB', 'GRAY']:
                    fw.write('import numpy as np\n')
                    fw.write('import cv2\n\n')
                    fw.write('def image_preprocess(img_path, norm=True):\n')
                    fw.write('    img = cv2.imread(img_path)\n')
                    fw.write('    if img is None:\n')
                    fw.write('        raise FileNotFoundError(\'No such image: {}\'.format(img_path))\n')
                    H = model_inputs[idx]['shape'][1]
                    W = model_inputs[idx]['shape'][2]
                    fw.write('    img = cv2.resize(img, ({}, {}), interpolation=cv2.INTER_LINEAR)\n'.format(H, W))
                    if training_input_formats[idx] in ['BGR', 'RGB']:
                        if training_input_formats[idx] == 'RGB' and input_formats[idx] == 'GRAY':
                            if calibrator_custom.utils.get_sdk_version() in ['1', 'Q_0']:
                                fw.write('    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n')
                                fw.write('    if norm:\n')
                                fw.write('        img = (img - {}) / {}\n'.format(mean[idx][0], std[idx][0]))
                                fw.write('        dummy = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)\n')
                                fw.write('        img = np.expand_dims(img, axis=2)\n')
                                fw.write('        img = np.concatenate([img, dummy], axis=-1)\n')
                                fw.write('        img = img.astype(np.float32)\n\n')
                                fw.write('    return np.expand_dims(img, 0)\n')
                            else:
                                fw.write('    if norm:\n')
                                fw.write('        img = (img - {}) / {}\n'.format(
                                    mean[idx][0], std[idx][0]))
                                fw.write('        img = img.astype(np.float32)\n')
                                fw.write('    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n')
                                fw.write('    img = np.expand_dims(img, 2)\n\n')
                                fw.write('    return np.expand_dims(img, 0)\n')
                        else:
                            fw.write('    if norm:\n')
                            fw.write('        img = (img - [{}, {}, {}]) / [{}]\n'.format(
                                mean[idx][2], mean[idx][1], mean[idx][0], ', '.join(str(s) for s in std[idx])))
                            fw.write('        img = img.astype(np.float32)\n')
                            if training_input_formats[idx] == 'RGB':
                                fw.write('    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n\n')
                                fw.write('    return np.expand_dims(img, 0)\n')
                            else:
                                fw.write('\n')
                                fw.write('    return np.expand_dims(img, 0)\n')
                    else:
                        fw.write('    if norm:\n')
                        fw.write('        img = (img - {}) / {}\n'.format(
                            mean[idx][0], std[idx][0]))
                        fw.write('        img = img.astype(np.float32)\n')
                        fw.write('    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n')
                        fw.write('    img = np.expand_dims(img, 2)\n\n')
                        fw.write('    return np.expand_dims(img, 0)\n')

                elif 'RAWDATA' in training_input_formats[idx]:
                    fw.write('import numpy as np\n\n')
                    fw.write('def image_preprocess(img_path, norm=True):\n')
                    fw.write('    img = np.load(img_path)\n\n')
                    fw.write('    img = img.astype("float32")\n\n')
                    fw.write('    return img\n')

                else:
                    raise NotImplementedError('Not support auto generate preprocess for ({})'.format(
                        training_input_formats[idx]))

            print('{}({}) preprocess generated in {}'.format(input_arrays[idx], idx, file_name))
            model_preprocess['preprocess_script'] = file_name
            preprocess_scripts.append(model_preprocess)

        else:
            raise ValueError('Model input({}): {} dosen\'t match input_config inputs({}): {}'.format(
                idx, model_inputs[idx]['name'], idx, input_arrays[idx]))

    return preprocess_scripts


def chalk_image_list(inputs):
    if ':' in inputs:
        dir_name = inputs.split(':')[0]
        base_name = inputs.split(':')[-1]
    else:
        dir_name = None
        base_name = inputs
    if not os.path.exists(base_name):
        raise FileNotFoundError('No such {} image or directory.'.format(base_name))
    if os.path.isdir(base_name):
        image_list = utils.all_path(base_name)
    elif os.path.basename(base_name).split('.')[-1].lower() in utils.image_suffix:
        image_list = [base_name]
    else:
        with open(base_name, 'r') as f:
            multi_images = f.readlines()
        if dir_name is None:
            image_list = [images.strip().split(',') for images in multi_images]
        else:
            image_list = [[os.path.join(dir_name, i) for i in images.strip().split(',')] for images in multi_images]

    return image_list


def chalk_image_generate(inputs, preprocess_scripts):
    preprocess_funcs = [utils.image_preprocess_func(pre) for pre in preprocess_scripts]
    image_list = chalk_image_list(inputs)
    img_gen = utils.image_generator(image_list, preprocess_funcs)

    return img_gen
