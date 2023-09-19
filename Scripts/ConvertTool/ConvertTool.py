# -*- coding: utf-8 -*-

import shutil
import os
import argparse
import pdb
import sys
import subprocess
import tensorflow as tf
import calibrator_custom
import json
import pickle
import six
from calibrator_custom import utils
from caffe_convert_tool import convert_from_caffemodel as _convert_from_caffemodel
from onnx_convert_tool import convert_from_onnx
from mace.python.tools.convert_util import mace_check, setAutogenWarning, getAutogenIniPath
from mace.python.tools.sgs_onnx import SGSModel_transform_onnx
from mace.python.tools.sgs_onnx import SGSModel_transform_onnx_S
from tflite_convert_tool import convert_from_tflite
import configparser
import numpy as np
import string
import argparse
import importlib
import os
import sys
import configparser
import six



if 'IPU_TOOL' in os.environ:
    Project_path = os.environ['IPU_TOOL']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/postprocess"))
elif 'SGS_IPU_DIR' in os.environ:
    Project_path = os.environ['SGS_IPU_DIR']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/postprocess"))
else:
    raise OSError('Run `source cfg_env.sh` in top directory.')

def arg_parse():
    parser = argparse.ArgumentParser(description='Convert Tool')

    subparsers = parser.add_subparsers(help='platform info')

    #tensorflow graphdef
    graphdef_parser = subparsers.add_parser('tensorflow_graphdef', help='tensorflow graphdef commands')
    graphdef_parser.add_argument('--graph_def_file', type=str, required=True,
                        help='Full filepath of file containing frozen GraphDef')
    graphdef_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    graphdef_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    graphdef_parser.add_argument('--input_arrays', type=str, required=False,
                        help='Names of the input arrays, comma-separated.')
    graphdef_parser.add_argument('--output_arrays', type=str, required=False,
                        help='Names of the output arrays, comma-separated.')
    graphdef_parser.add_argument('--input_shapes', type=str, required=False, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form batch size, input array height, input array width, input array depth. (default None)')
    graphdef_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    graphdef_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    graphdef_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    graphdef_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')

    if 'IPU_TOOL' in os.environ:
        graphdef_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #tensorflow save_model
    savemodel_parser = subparsers.add_parser('tensorflow_savemodel', help='tensorflow save_model commands')
    savemodel_parser.add_argument('--saved_model_dir', type=str, required=True,
                        help='SavedModel directory to convert')
    savemodel_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    savemodel_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    savemodel_parser.add_argument('--input_arrays', type=str, required=False, default=None,
                        help='Names of the input arrays, comma-separated.')
    savemodel_parser.add_argument('--input_shapes', type=str, required=False, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form batch size, input array height, input array width, input array depth. (default None)')
    savemodel_parser.add_argument('--output_arrays', type=str, required=False, default=None,
                        help='Names of the output arrays, comma-separated.')
    savemodel_parser.add_argument('--tag_set', type=str, required=False, default=None,
                        help='Set of tags identifying the MetaGraphDef within the SavedModel to analyze. All tags in the tag set must be present. (default None)')
    savemodel_parser.add_argument('--signature_key', type=str, required=False, default=None,
                        help='Key identifying SignatureDef containing inputs and outputs.(default DEFAULT_SERVING_SIGNATURE_DEF_KEY)')

    savemodel_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    savemodel_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    savemodel_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    savemodel_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')

    if 'IPU_TOOL' in os.environ:
        savemodel_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #keras h5
    keras_parser = subparsers.add_parser('keras', help='keras commands')
    keras_parser.add_argument('--model_file', type=str, required=True,
                        help='Full filepath of HDF5 file containing the tf.keras model.')
    keras_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    keras_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    keras_parser.add_argument('--input_arrays', type=str, required=False, default=None,
                        help='Names of the input arrays, comma-separated. (default None).')
    keras_parser.add_argument('--input_shapes', type=str, required=False, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form batch size, input array height, input array width, input array depth. (default None)')
    keras_parser.add_argument('--output_arrays', type=str, required=False, default=None,
                        help='Names of the output arrays, comma-separated. (default None)')
    keras_parser.add_argument('--custom_objects', type=str, required=False, default=None,
                        help='Dict mapping names (strings) to custom classes or functions to be considered during model deserialization. (default None)')

    keras_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    keras_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    keras_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    keras_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')
    if 'IPU_TOOL' in os.environ:
        keras_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #tensorflow lite
    tflite_parser = subparsers.add_parser('tflite', help='tflite commands')
    tflite_parser.add_argument('--model_file', type=str, required=True,
                        help='Full filepath of tflite file containing the tflite model.')
    tflite_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    tflite_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    tflite_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    tflite_parser.add_argument('--fixed_model', type=str, default=False,
                        help='If tflite is fixed_model, parser the tensor quantizaiton parameters')
    tflite_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    tflite_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    tflite_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')
    if 'IPU_TOOL' in os.environ:
        tflite_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #caffe model
    caffe_parser = subparsers.add_parser('caffe', help='caffe commands')
    caffe_parser.add_argument('--model_file', type=str, required=True,
                        help='Full filepath of tflite file containing the caffe model.')
    caffe_parser.add_argument('--weight_file', type=str, required=True,
                        help='Full filepath of tflite file containing the caffe weight.')
    caffe_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    caffe_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    caffe_parser.add_argument('--input_arrays', type=str, required=False, default=None,
                        help='Names of the input arrays, comma-separated. (default None).')
    caffe_parser.add_argument('--output_arrays', type=str, required=False, default=None,
                        help='Names of the output arrays, comma-separated. (default None)')
    caffe_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    caffe_parser.add_argument('--output_pack_model_arrays', type=str, required=False, default='None',
                        help='Set output Pack model, specify name pack model like caffe(NCHW),comma-separated. All outputTersors will be NCHW if set "caffe" (default is NHWC)')
    caffe_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    caffe_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    caffe_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')
    if 'IPU_TOOL' in os.environ:
        caffe_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #onnx model
    onnx_parser = subparsers.add_parser('onnx', help='onnx commands')
    onnx_parser.add_argument('--model_file', type=str, required=True,
                        help='Full filepath of tflite file containing the onnx model.')
    onnx_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    onnx_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    onnx_parser.add_argument('--input_shapes', type=str, required=True, default=None,
                            help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form N C H W (default None)')
    onnx_parser.add_argument('--input_arrays', type=str, required=False, default=None,
                        help='Names of the input arrays, comma-separated. (default None).')
    onnx_parser.add_argument('--output_arrays', type=str, required=False, default=None,
                        help='Names of the output arrays, comma-separated. (default None)')
    onnx_parser.add_argument('--fixedC2fixedWO', required=False, default=False, action='store_true',
                        help='If onnx is fixed, we transform onnx fixed to sgs fixed model.')
    onnx_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    onnx_parser.add_argument('--output_pack_model_arrays', type=str, required=False, default='None',
                        help='output tensor pack model,NCHW(caffe pytorch) or NHWC(tensorflow). All inputTersors will be NCHW if set "onnx",default NHWC')
    onnx_parser.add_argument('--skip_simplify', default=False, action='store_true',
                        help='Skip onnxsim simplify ONNX model.')
    onnx_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    onnx_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    onnx_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')
    if 'IPU_TOOL' in os.environ:
        onnx_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    # Debug.sim
    if 'IPU_TOOL' in os.environ:
        debug_parser = subparsers.add_parser('debug', help='debug commands')
        debug_parser.add_argument('--model_file', type=str, required=True,
                            help='Full filepath of tflite file containing the Debug model.')
        debug_parser.add_argument('--input_config', type=str, required=True,
                            help='Input config path.')
        debug_parser.add_argument('--output_file', type=str, required=True,
                            help='Full filepath of out Model path.')
        debug_parser.add_argument('--quant_file', type=str, default=None,
                            help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
        debug_parser.add_argument('--show_log', default=False, action='store_true',
                            help='Show log on screen.')
        debug_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
        debug_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')
        debug_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')
        debug_parser.add_argument('--convert_kb', default=False, action='store_true',
                            help='Convert kb const tensor')

    return parser.parse_args()

def read_args_from_ini(input_config):
    # 1.create ini class
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(input_config, encoding='utf-8')

    # 1.read '--input_arryas' and '--output_arrays' from ini
    input_arrays_str = config['INPUT_CONFIG']['inputs'].replace(" ","")
    input_arrays_str = input_arrays_str.strip(string.punctuation)
    output_arrays_str = config['OUTPUT_CONFIG']['outputs'].replace(" ","")
    output_arrays_str = output_arrays_str.strip(string.punctuation)
    SGSModel_transform_onnx.INPUT_CONFIG_INI = input_config

    # 2.read'--input_pack_model_arrays' from ini
    add_output_pack_inINI = False
    if 'input_layouts' in config['INPUT_CONFIG']:
        list_input_pack_model_arrays = config['INPUT_CONFIG']['input_layouts'].strip(string.punctuation).replace(" ","") # return list
        str_input_pack_model_arrays = ''.join(list_input_pack_model_arrays)
        input_pack_model_arrays = str_input_pack_model_arrays
    else:
        input_pack_model_arrays = 'None'

    # 3.read'--output_pack_model_arrays' from ini
    add_output_pack_inINI = False
    if 'output_formats' in config['OUTPUT_CONFIG']:
        print('required args: -- output_pack_model_arrays is True')
        list_output_pack_model_arrays = config['OUTPUT_CONFIG']['output_formats'].strip(string.punctuation).replace(" ","") # return list
        str_output_pack_model_arrays = ''.join(list_output_pack_model_arrays)
        output_pack_model_arrays = str_output_pack_model_arrays
    elif 'output_layouts' in config['OUTPUT_CONFIG']:
        print('required args: -- output_pack_model_arrays is True')
        list_output_pack_model_arrays = config['OUTPUT_CONFIG']['output_layouts'].strip(string.punctuation).replace(" ","") # return list
        str_output_pack_model_arrays = ''.join(list_output_pack_model_arrays)
        output_pack_model_arrays = str_output_pack_model_arrays
    else:
        add_output_pack_inINI = True
        output_pack_model_arrays = 'None'
    # '--output_pack_model_arrays': cmd >ini
    platform = sys.argv[1]
    args = arg_parse()
    if (platform == 'caffe' and args.output_pack_model_arrays != 'None') or (platform == 'onnx' and args.output_pack_model_arrays != 'None'):
        list_output_pack_model_arrays = args.output_pack_model_arrays.split(',')
        for i in six.moves.range(len(list_output_pack_model_arrays)):
            if list_output_pack_model_arrays[i] == platform:
                list_output_pack_model_arrays[i] = 'NCHW'
        str_output_pack_model_arrays = ','.join(list_output_pack_model_arrays)
        output_pack_model_arrays = str_output_pack_model_arrays

        list_output_arrays_str = output_arrays_str.split(',')
        output_index = []
        for i in six.moves.range(len(list_output_arrays_str)):
            for j in six.moves.range(len(list_output_pack_model_arrays)):
                if list_output_arrays_str[i] == list_output_pack_model_arrays[j]:
                    list_output_pack_model_arrays[j] = 'NCHW'
                    output_index.append(i)
        if len(list_output_arrays_str) == len(list_output_pack_model_arrays):
            output_pack_model_arrays = ','.join(list_output_pack_model_arrays)
        elif len(list_output_arrays_str) != len(list_output_pack_model_arrays):
            new_list_output_pack_model_arrays = []
            for i in six.moves.range(len(list_output_arrays_str)):
                if i in output_index:
                    new_list_output_pack_model_arrays.append('NCHW')
                else:
                    new_list_output_pack_model_arrays.append('NHWC')
            str_output_pack_model_arrays = ','.join(new_list_output_pack_model_arrays)
            output_pack_model_arrays = str_output_pack_model_arrays

    # add 'output_layouts' in ini if there is no 'output_layouts' written in ini
    if platform == 'caffe' or platform == 'onnx':
        if add_output_pack_inINI == True and args.output_pack_model_arrays != 'None':
            list_output_pack_model_arrays = output_pack_model_arrays.split(',')
            for i in six.moves.range(len(list_output_pack_model_arrays)):
                if list_output_pack_model_arrays[i] == 'None':
                    list_output_pack_model_arrays[i] = 'NHWC'
            output_pack_model_arrays = ','.join(list_output_pack_model_arrays)
            config['OUTPUT_CONFIG']['output_layouts'] = output_pack_model_arrays + ';'
            with open(input_config, mode='w', encoding='utf-8', errors='ignore') as f:
                config.write(f)

    ## 4.read'--quantizations' from ini
    add_quantizations_inINI = False
    if 'quantizations'  not in config['INPUT_CONFIG']:
        add_quantizations_inINI = True
        quantizations_list = []
        list_input_arrays = input_arrays_str.split(',')
        for i in six.moves.range(len(list_input_arrays)):
            quantizations_list.append('TRUE')
        quantizations_arrays = ','.join(quantizations_list)
    if add_quantizations_inINI == True:
        config['INPUT_CONFIG']['quantizations'] = quantizations_arrays + ';'
        with open(input_config, mode='w', encoding='utf-8', errors='ignore') as f:
            config.write(f)

    ## 5.read'--dequantizations' from ini
    add_dequantizations_inINI = False
    if 'dequantizations' not in config['OUTPUT_CONFIG']:
        add_dequantizations_inINI = True
        dequantizations_list = []
        list_output_arrays = output_arrays_str.split(',')
        for i in six.moves.range(len(list_output_arrays)):
            dequantizations_list.append('TRUE')
        dequantizations_arrays = ','.join(dequantizations_list)
    if add_dequantizations_inINI == True:
        config['OUTPUT_CONFIG']['dequantizations'] = dequantizations_arrays + ';'
        with open(input_config, mode='w', encoding='utf-8', errors='ignore') as f:
            config.write(f)

    return input_arrays_str, output_arrays_str, input_pack_model_arrays, output_pack_model_arrays, input_config

def clear_directory(input_config):

    def remove_folder(folder_name):
        if os.path.exists(folder_name):
            if os.path.isdir(folder_name):
                shutil.rmtree(folder_name)
            else:
                os.remove(folder_name)

    remove_folder('lstm_data')

    autogen_path = os.path.join(os.path.dirname(input_config),
                                'autogen_' + os.path.basename(input_config))
    if os.path.exists(autogen_path):
        os.remove(autogen_path)
platform_buffer_map = {}

def SGS_Convert(caffe_src_buf=None, caffe_weight_buf=None, onnx_buf=None):
    caffe_buffer_dic = {}
    caffe_buffer_dic['caffe_src'] = caffe_src_buf
    caffe_buffer_dic['caffe_weight'] = caffe_weight_buf
    platform_buffer_map['caffe'] = caffe_buffer_dic
    platform_buffer_map['onnx'] = onnx_buf
    main()

def main():
    platform = sys.argv[1]
    args = arg_parse()
    input_config = args.input_config
    output_file = args.output_file
    quant_file = args.quant_file
    fixedC2fixedWO = False
    lstm_flag = False
    gru_flag = False
    postfile_list = []
    lstm_postfile_list = []
    gru_postfile_list = []
    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, 'Converted_Net_float.sim')
    elif (os.path.isdir(os.path.dirname(output_file)) or (os.path.dirname(output_file) == '')) :
        pass
    else:
        raise OSError('\033[31mCould not access path: {}\033[0m'.format(output_file))

    mace_check(os.path.exists(input_config),'can not find ini file')
    input_name_format_map = {}
    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read(input_config, encoding='utf-8')
    training_input_formats,input_formats = 'None','None'
    try:
        input_formats = conf.get('INPUT_CONFIG', 'input_formats')
    except:
        training_input_formats = conf.get('INPUT_CONFIG', 'training_input_formats')
    real_input_format = input_formats if input_formats != 'None' else training_input_formats

    input_arrays_str, output_arrays_str, input_pack_model_arrays, output_pack_model_arrays, input_config = read_args_from_ini(input_config)
    SGSModel_transform_onnx.INPUT_CONFIG_INI = input_config
    SGSModel_transform_onnx_S.INPUT_CONFIG_INI = input_config

    clear_directory(input_config)

    if platform != 'tflite':
        input_arrays = [input_str for input_str in input_arrays_str.split(',')]
        input_format_array = real_input_format.replace(";","").split(',')
        #compitable lstm case
        #mace_check(len(input_arrays) == len(input_format_array),'input num != input format num')
        for i in range(len(input_arrays)):
            input_name_format_map[input_arrays[i]] = input_format_array[i]

    if platform == 'tensorflow_graphdef':
#        if 'tensorflow_graphdef' in platform_buffer_map:
#            if platform_buffer_map['tensorflow_graphdef'] != None:
#                graph_def_file = platform_buffer_map['tensorflow_graphdef']
#            else:
#                graph_def_file = args.graph_def_file
#        else:
#            graph_def_file = args.graph_def_file
        graph_def_file = args.graph_def_file
        input_arrays = input_arrays_str
        output_arrays = output_arrays_str

        input_shapes_str = args.input_shapes
        input_arrays = [input_str for input_str in input_arrays_str.split(',')]
        output_arrays = [output_str for output_str in output_arrays_str.split(',')]
        input_shapes_str_list = input_shapes_str.split(':') if input_shapes_str is not None else None
        if input_arrays is not None and input_shapes_str_list is not None and len(input_arrays) != len(input_shapes_str_list):
            raise NameError('input_arrays\'s lengh is not equal input_shaps.')
        if (input_shapes_str_list is not None):
            input_shapes = {
                    input_arrays[t_num]: [int(i) for i in input_shapes_str_list[t_num].split(',')] for t_num in range(len(input_arrays))
            }
        else:
            input_shapes = None
        #convert to tflite model
        tfliteModel = tf.lite.TFLiteConverter.from_frozen_graph(
                graph_def_file, input_arrays, output_arrays, input_shapes).convert()
        tfliteModel_output_path = output_file.strip().split('.sim')[0]+ '.tflite'
        open(tfliteModel_output_path, "wb").write(tfliteModel)
        #tflite convert to mace
        model_file = tfliteModel_output_path
        output_file = args.output_file
        converter = convert_from_tflite(model_file ,output_file)

    elif platform == 'tensorflow_savemodel':
#        if 'tensorflow_savemodel' in platform_buffer_map:
#            if platform_buffer_map['tensorflow_savemodel'] != None:
#                saved_model_dir = platform_buffer_map['tensorflow_savemodel']
#            else:
#                saved_model_dir = args.saved_model_dir
#        else:
#            saved_model_dir = args.saved_model_dir
        saved_model_dir = args.saved_model_dir
        input_arrays = input_arrays_str
        output_arrays = output_arrays_str

        input_shapes_str = args.input_shapes
        tag_set = args.tag_set
        tag_set = set([tag_set]) if tag_set is not None else None
        signature_key = args.signature_key

        input_arrays = [input_str for input_str in input_arrays_str.split(',')]  if input_arrays_str is not None else None
        output_arrays = [output_str for output_str in output_arrays_str.split(',')] if output_arrays_str is not None else None
        input_shapes_str_list = input_shapes_str.split(':') if input_shapes_str is not None else None

        if input_arrays is not None and input_shapes_str_list is not None and len(input_arrays) != len(input_shapes_str_list):
            raise NameError('input_arrays\'s lengh is not equal input_shaps.')

        if (input_shapes_str_list is not None):
            input_shapes = {
                    input_arrays[t_num]: [int(i) for i in input_shapes_str_list[t_num].split(',')] for t_num in range(len(input_arrays))
            }
        else:
            input_shapes = None
        tfliteModel = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir,
                                                             input_arrays,
                                                             input_shapes,
                                                             output_arrays,
                                                             tag_set,
                                                             signature_key)
        tfliteModel_output_path = output_file.strip().split('.sim')[0]+ '.tflite'
        open(tfliteModel_output_path, "wb").write(tfliteModel)
        #tflite convert to mace
        model_file = tfliteModel_output_path
        output_file = args.output_file
        converter = convert_from_tflite(model_file ,output_file)

    elif platform == 'keras':
        model_file = args.model_file
        input_arrays = input_arrays_str
        output_arrays = output_arrays_str

        input_shapes_str = args.input_shapes
        custom_objects = args.custom_objects

        input_arrays = [input_str for input_str in input_arrays_str.split(',')]  if input_arrays_str is not None else None
        output_arrays = [output_str for output_str in output_arrays_str.split(',')] if output_arrays_str is not None else None
        input_shapes_str_list = input_shapes_str.split(':') if input_shapes_str is not None else None

        if input_arrays is not None and input_shapes_str_list is not None and len(input_arrays) != len(input_shapes_str_list):
            raise NameError('input_arrays\'s lengh is not equal input_shaps.')

        if (input_shapes_str_list is not None):
            input_shapes = {
                    input_arrays[t_num]: [int(i) for i in input_shapes_str_list[t_num].split(',')] for t_num in range(len(input_arrays))
            }
        else:
            input_shapes = None

        tfliteModel = tf.lite.TFLiteConverter.from_keras_model_file(model_file,
                                                                  input_arrays,
                                                                  input_shapes,
                                                                  output_arrays,
                                                                  custom_objects)
        tfliteModel_output_path = output_file.strip().split('.sim')[0]+ '.tflite'
        open(tfliteModel_output_path, "wb").write(tfliteModel)
        #tflite convert to mace
        model_file = tfliteModel_output_path
        output_file = args.output_file
        converter = convert_from_tflite(model_file ,output_file)

    elif platform == 'tflite':
        model_file = args.model_file
        input_config = args.input_config
        input_arrays = [input_str for input_str in input_arrays_str.split(',')]  if input_arrays_str is not None else None
        output_arrays = [output_str for output_str in output_arrays_str.split(',')] if output_arrays_str is not None else None
        bFixed_model = args.fixed_model
        converter = convert_from_tflite(model_file, output_file, bFixed_model)

    elif platform == 'caffe':
        if 'caffe' in platform_buffer_map:
            if platform_buffer_map['caffe']['caffe_src'] != None and platform_buffer_map['caffe']['caffe_weight'] != None:
                model_file = platform_buffer_map['caffe']['caffe_src']
                weight_file = platform_buffer_map['caffe']['caffe_weight']
            else:
                model_file = args.model_file
                weight_file = args.weight_file
        else:
            model_file = args.model_file
            weight_file = args.weight_file

        input_arrays = input_arrays_str
        output_arrays = output_arrays_str

        input_pack_mode_arrays = input_pack_model_arrays
        output_pack_mode_arrays = output_pack_model_arrays

        converter = _convert_from_caffemodel(model_file ,
                                             weight_file,
                                             input_arrays,
                                             output_arrays,
                                             output_file,
                                             input_pack_mode_arrays,
                                             output_pack_mode_arrays,
                                             input_name_format_map)

    elif platform == 'onnx':
        if 'onnx' in platform_buffer_map:
            if platform_buffer_map['onnx'] != None:
                model_file = platform_buffer_map['onnx']
            else:
                model_file = args.model_file
        else:
            model_file = args.model_file

        model_file = args.model_file
        input_arrays = input_arrays_str
        output_arrays = output_arrays_str

        input_shapes = args.input_shapes
        input_pack_model = input_pack_model_arrays
        output_pack_model = output_pack_model_arrays
        fixedC2fixedWO = args.fixedC2fixedWO
        skip_simplify = args.skip_simplify
        converter = convert_from_onnx(model_file ,
                                             input_arrays,
                                             input_shapes,
                                             output_arrays,
                                             output_file,
                                             input_pack_model,
                                             output_pack_model,
                                             input_name_format_map,
                                             fixedC2fixedWO,
                                             skip_simplify)

    if platform == 'debug':
        model_file = args.model_file
        output_file = args.output_file
        input_config = args.input_config
        if args.convert_kb:
           convert_kb = True
           converter = convert_from_tflite(model_file,
                                        output_file,convert_kb=convert_kb
                                        )
           debug_output = os.path.basename(output_file)
           debug_output = output_file.replace(debug_output, 'Debug_' + debug_output)
           tflite_model = converter.convert()
           lstm_flag = True if converter._lstm_num > 0 else False
           gru_flag = True if converter._gru_num > 0 else False
           with open(debug_output, "wb") as f:
                f.write(tflite_model)

        else:
            debug_output = args.model_file
    else:
        debug_output = os.path.basename(output_file)
        debug_output = output_file.replace(debug_output, 'Debug_' + debug_output)
        tflite_model = converter.convert()
        lstm_flag = True if converter._lstm_num > 0 else False
        gru_flag = True if converter._gru_num > 0 else False
        with open(debug_output, "wb") as f:
            f.write(tflite_model)

    if lstm_flag or gru_flag:
        if 'IPU_TOOL' in os.environ:
            Project_path = os.environ['IPU_TOOL']
        elif 'SGS_IPU_DIR' in os.environ:
            Project_path = os.environ['SGS_IPU_DIR']
        else:
            raise OSError('Run `source cfg_env.sh` in top directory.')
        if lstm_flag:
            if platform == 'onnx':
                postprocess_file = os.path.join(Project_path,'Scripts/postprocess/sgs_chalk_postprocess_method/onnx_lstm_unroll.py')
            else:
                postprocess_file = os.path.join(Project_path,'Scripts/postprocess/sgs_chalk_postprocess_method/caffe_lstm_unroll.py')

            if os.path.exists(postprocess_file) and postprocess_file.split('.')[-1] == 'py':
                sys.path.append(os.path.dirname(postprocess_file))
                postprocess_func = importlib.import_module(os.path.basename(postprocess_file).split('.')[0])
                lstm_postfile_list = postprocess_func.model_postprocess()
        elif gru_flag:
            if platform == 'onnx':
                postprocess_file = os.path.join(Project_path,'Scripts/postprocess/sgs_chalk_postprocess_method/onnx_gru_unroll.py')
            else:
                raise ValueError('no support platform')

            if os.path.exists(postprocess_file) and postprocess_file.split('.')[-1] == 'py':
                sys.path.append(os.path.dirname(postprocess_file))
                postprocess_func = importlib.import_module(os.path.basename(postprocess_file).split('.')[0])
                gru_postfile_list = postprocess_func.model_postprocess()
        else:
            #postfile_list = eval(postprocess_file).model_postprocess()
            raise ValueError('please input postprocess file with full path')
        postfile_list.extend(lstm_postfile_list)
        postfile_list.extend(gru_postfile_list)
        model_file = debug_output
        converter = convert_from_tflite(model_file,
                                       output_file,
                                       postfile_list,
                                       'concat',
                                       )

        debug_output = os.path.basename(output_file)
        debug_output = output_file.replace(debug_output, 'Concat_' + debug_output)
        concat_model = converter.convert()
        with open(debug_output, "wb") as f:
            f.write(concat_model)

    if args.postprocess is not None:
        postprocess_file = args.postprocess
        if os.path.exists(postprocess_file) and postprocess_file.split('.')[-1] == 'py':
            sys.path.append(os.path.dirname(postprocess_file))
            postprocess_func = importlib.import_module(os.path.basename(postprocess_file).split('.')[0])
            postfile_list = postprocess_func.model_postprocess()
        else:
            #postfile_list = eval(postprocess_file).model_postprocess()
            raise ValueError('please input postprocess file with full path')
        postfile_list = postfile_list[0] if isinstance(postfile_list,tuple) and len(postfile_list)==2 else postfile_list

        # mode == 'append'
        if args.postprocess_input_config is None:
            config = configparser.ConfigParser()
            input_config = args.input_config
            # auto generate autogen_input_config.ini
            autogen_path = getAutogenIniPath(input_config, config)

            from mace.python.tools.converter_tool import tflite_converter

            postprocess_converter = tflite_converter.TfliteConverter(postfile_list,bFixed_model=False)
            tflite2mace_graph_def_post, list_input_name_post, list_output_name_post = postprocess_converter.run()
            copy_num = len(list_output_name_post)

            # change 'input_config.ini' content
            for key in config['OUTPUT_CONFIG']:
                if key == 'outputs':
                    inputs_value = ','.join(list_output_name_post) + ';'
                    config.set('OUTPUT_CONFIG','outputs',inputs_value)
                elif key == 'dequantizations':
                    dequantizations_result = ''
                    for i in six.moves.range(copy_num):
                        dequantizations_result = dequantizations_result + 'TRUE' + ','
                    dequantizations_result = dequantizations_result.rstrip(',')
                    config.set('OUTPUT_CONFIG','dequantizations',dequantizations_result)
            with open(autogen_path, mode='w', encoding='utf-8', errors='ignore') as f:
                config.write(f)
            setAutogenWarning(autogen_path)
        else:
            input_config = args.postprocess_input_config
        postfile_list = [postfile_list]

        model_file = debug_output
        converter = convert_from_tflite(model_file,
                                       output_file,
                                       postfile_list,
                                       'append',
                                       )

        debug_output = os.path.basename(output_file)
        debug_output = output_file.replace(debug_output, 'Concat_' + debug_output)
        concat_model = converter.convert()
        with open(debug_output, "wb") as f:
            f.write(concat_model)


    if 'IPU_TOOL' in os.environ and args.to_debug:
        print('\nDebug model at: %s\n' % (debug_output))
        return


    model_converter = calibrator_custom.converter(debug_output, input_config, show_log=args.show_log)
    compiler_config = calibrator_custom.utils.CompilerConfig()

    if quant_file is not None:
        quant_param = None
        if not os.path.exists(quant_file):
            raise FileNotFoundError('No such quant_file: {}'.format(quant_file))
        else:
            try:
                with open(quant_file, 'rb') as f:
                    quant_param = pickle.load(f)
            except pickle.UnpicklingError:
                with open(quant_file, 'r') as f:
                    quant_param = json.load(f)
            except json.JSONDecodeError:
                raise ValueError('quant_param only support JSON or Pickle file.')
        calibrator_custom.utils.check_quant_param(quant_param)
        model_converter.update(quant_param)
        model_converter.convert(compiler_config.Debug2FloatConfig(SkipInfectDtype=True), saved_path=output_file)
        print('\nFloat model at: %s\n' % (output_file))
    else:
        if fixedC2fixedWO:
            model_converter.convert(compiler_config.Float2FixedWOConfig(), saved_path=output_file, model_type='Fixed_without_ipu_ctrl')
            print('\nFixed model at: %s\n' % (output_file))
        else:
            model_converter.convert(compiler_config.Debug2FloatConfig(), saved_path=output_file, model_type='Float')
            print('\nFloat model at: %s\n' % (output_file))


if __name__ == '__main__':
    main()
