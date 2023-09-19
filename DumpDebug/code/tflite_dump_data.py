# -*- coding: utf-8 -*-

import os
import sys
import tqdm
import struct
import shutil
import argparse
import numpy as np
import tensorflow as tf
from functools import reduce
from calibrator_custom import utils

if 'SGS_IPU_DIR' in os.environ:
    Project_path = os.environ['SGS_IPU_DIR']
    sys.path.append(os.path.join(Project_path, 'Scripts/ConvertTool/third_party'))
elif 'IPU_TOOL' in os.environ:
    Project_path = os.environ['IPU_TOOL']
    sys.path.append(os.path.join(Project_path, 'Scripts/ConvertTool/third_party'))
else:
    raise OSError('Run source cfg_env.sh in top directory.')

import tflite


def arg_parse():
    parser = argparse.ArgumentParser(description='Simulator Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='TFLite Model path.')
    parser.add_argument('--dump_bin', default='True', type=str, choices=['True', 'False'],
                        help='Dump data save as binary (or string). True or False')
    parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')

    return parser.parse_args()


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


def write_txt(f, data):
    for num, value in enumerate(list(data.flat)):
        f.write('{:.6f}, '.format(value).encode('ascii'))
        if (num + 1) % 16 == 0:
            f.write(b'\n')
    if len(list(data.flat)) % 16 != 0:
        f.write(b'\n')


def convert_bytes(data):
    data_list = data.astype(np.float32).flatten().tolist()
    return struct.pack('f' * len(data_list), *data_list)


def save_results(output_info, output_data, dump_bin):
    with open('dumpData/tflite_outtensor_dump.bin', 'ab') as f:
        shapes = ', '.join([str(i) for i in output_info['shape'].tolist()])
        line0 = '//out {} s: 0.000000 z: 0 type: float name: {} bConstant:0 shape:[{}] dims:{}\n'.format(
            output_info['output_idx'], output_info['name'], shapes, len(output_info['shape'].tolist()))
        f.write(line0.encode('ascii'))
        line1 = 'op_out[{}] {} = {{\n'.format(output_info['op_index'], output_info['op_name'])
        f.write(line1.encode('ascii'))
        line2 = '//buffer data size: {}\n'.format(reduce(lambda x, y: x * y, output_info['shape'].tolist()) * 4)
        f.write(line2.encode('ascii'))
        if dump_bin == 'True':
            f.write(convert_bytes(output_data))
        else:
            write_txt(f, output_data)
        line3 = '};\n'
        f.write(line3.encode('ascii'))


def prepare_input_data(model_buf, image_path, preprocess):
    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        dir_name = None
        base_name = image_path
    if not os.path.exists(base_name):
        raise FileNotFoundError('No such {} image or directory.'.format(base_name))

    preprocess_funcs = [utils.image_preprocess_func(n) for n in preprocess.split(',')]
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
    if len(image_list) > 1:
        print('[WARRING] Dump tensor only need 1 input for model, choose {} for input.'.format(image_list[0]))

    infer = tf.lite.Interpreter(model_content=model_buf)
    model_inputs = infer.get_input_details()
    num_input_images = len(image_list[0]) if isinstance(image_list[0], list) else 1
    if len(preprocess_funcs) != len(model_inputs) or len(model_inputs) != num_input_images:
        raise ValueError('Can not set_input, model has {} inputs, but got {} inputs and {} preprocess_methods!'.format(
            len(model_inputs), num_input_images, len(preprocess_funcs)))

    input_data = []
    for idx, preprocess_func in enumerate(preprocess_funcs):
        if isinstance(image_list[0], list):
            input_data.append(preprocess_func(image_list[0][idx], norm=True))
        else:
            return preprocess_func(image_list[0], norm=True)
    return input_data


def set_input(infer, input_data):
    model_inputs = infer.get_input_details()
    for idx, input_info in enumerate(model_inputs):
        if isinstance(input_data, list):
            infer.set_tensor(input_info['index'], input_data[idx])
        else:
            infer.set_tensor(input_info['index'], input_data)


# reference https://github.com/raymond-li/tflite_tensor_outputter
# For dump tflite tensor data
def buffer_change_output_tensor_to(model_buffer, new_tensor_i):

    fb_model_root = tflite.Model.Model.GetRootAsModel(model_buffer, 0)
    output_tensor_index_offset = fb_model_root.Subgraphs(0).OutputsOffset(0)

    # Flatbuffer scalars are stored in little-endian.
    new_tensor_i_bytes = bytes([
    new_tensor_i & 0x000000FF, \
    (new_tensor_i & 0x0000FF00) >> 8, \
    (new_tensor_i & 0x00FF0000) >> 16, \
    (new_tensor_i & 0xFF000000) >> 24 \
    ])
    # Replace the 4 bytes corresponding to the first output tensor index
    return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]


def getType(CLASS, code):
    for name, value in CLASS.__dict__.items():
        if value == code:
            return name
    return 'UNKNOWN'


def get_operator_outputs(model_buf):
    tf_infer = tf.lite.Interpreter(model_content=model_buf)
    tensors = tf_infer.get_tensor_details()
    operator_outputs = []
    model = tflite.Model.Model.GetRootAsModel(model_buf, 0)
    subgraph = model.Subgraphs(0)
    for op_idx in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(op_idx)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        op_name = getType(tflite.BuiltinOperator.BuiltinOperator,
            max(op_code.DeprecatedBuiltinCode(), op_code.BuiltinCode()))
        for out_idx in range(op.OutputsLength()):
            tensor = tensors[op.Outputs(out_idx)]
            tensor['op_name'] = op_name
            tensor['output_idx'] = out_idx
            tensor['op_index'] = op_idx
            operator_outputs.append(tensor)
    return operator_outputs


def run_tflite(image_path, model_path, preprocess, dump_bin):
    with open('dumpData/tflite_outtensor_dump.bin', 'w') as f:
        f.write('isFloat: 1\n')
    with open(model_path, 'rb') as f:
        model_buf = f.read()
    output_tensors = get_operator_outputs(model_buf)
    input_data = prepare_input_data(model_buf, image_path, preprocess)
    for out in tqdm.tqdm(output_tensors):
        modify_buf = buffer_change_output_tensor_to(model_buf, out['index'])
        infer = tf.lite.Interpreter(model_content=modify_buf)
        infer.allocate_tensors()
        set_input(infer, input_data)
        infer.invoke()
        output_data = infer.get_tensor(out['index'])
        save_results(out, output_data, dump_bin)
    print('File saved in dumpData/tflite_outtensor_dump.bin')


if __name__ == '__main__':
    args = arg_parse()
    image_path = args.image
    model_path = args.model
    preprocess = args.preprocess
    dump_bin = args.dump_bin

    if not os.path.exists(model_path):
        raise FileNotFoundError('{} not found'.format(model_path))

    renew_folder('dumpData')
    run_tflite(image_path, model_path, preprocess, dump_bin)
