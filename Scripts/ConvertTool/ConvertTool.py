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
from caffe_convert_tool import convert_from_caffemodel as _convert_from_caffemodel
from onnx_convert_tool import convert_from_onnx
import configparser

def arg_parse():
    parser = argparse.ArgumentParser(description='Convert Tool')

    subparsers = parser.add_subparsers(help='platform info')

    #tensorflow graphdef
    graphdef_parser = subparsers.add_parser('tensorflow_graphdef', help='tensorflow graphdef commands')
    graphdef_parser.add_argument('--graph_def_file', type=str, required=True,
                        help='Full filepath of file containing frozen GraphDef')
    graphdef_parser.add_argument('--input_arrays', type=str, required=True,
                        help='Names of the input arrays, comma-separated.')
    graphdef_parser.add_argument('--output_arrays', type=str, required=True,
                        help='Names of the output arrays, comma-separated.')
    graphdef_parser.add_argument('--input_shapes', type=str, required=False, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form batch size, input array height, input array width, input array depth. (default None)')
    graphdef_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    graphdef_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    graphdef_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        graphdef_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')

    #tensorflow save_model
    savemodel_parser = subparsers.add_parser('tensorflow_savemodel', help='tensorflow save_model commands')
    savemodel_parser.add_argument('--saved_model_dir', type=str, required=True,
                        help='SavedModel directory to convert')
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
    savemodel_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    savemodel_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    savemodel_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        savemodel_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')

    #keras h5
    keras_parser = subparsers.add_parser('keras', help='keras commands')
    keras_parser.add_argument('--model_file', type=str, required=True,
                        help='Full filepath of HDF5 file containing the tf.keras model.')
    keras_parser.add_argument('--input_arrays', type=str, required=False, default=None,
                        help='Names of the input arrays, comma-separated. (default None).')
    keras_parser.add_argument('--input_shapes', type=str, required=False, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form batch size, input array height, input array width, input array depth. (default None)')
    keras_parser.add_argument('--output_arrays', type=str, required=False, default=None,
                        help='Names of the output arrays, comma-separated. (default None)')
    keras_parser.add_argument('--custom_objects', type=str, required=False, default=None,
                        help='Dict mapping names (strings) to custom classes or functions to be considered during model deserialization. (default None)')
    keras_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    keras_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    keras_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        keras_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')

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
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        tflite_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')

    #caffe model
    caffe_parser = subparsers.add_parser('caffe', help='caffe commands')
    caffe_parser.add_argument('--model_file', type=str, required=True,
                        help='Full filepath of tflite file containing the caffe model.')
    caffe_parser.add_argument('--weight_file', type=str, required=True,
                        help='Full filepath of tflite file containing the caffe weight.')
    caffe_parser.add_argument('--input_arrays', type=str, required=False, default=None,
                        help='Names of the input arrays, comma-separated. (default None).')
    caffe_parser.add_argument('--output_arrays', type=str, required=False, default=None,
                        help='Names of the output arrays, comma-separated. (default None)')
    caffe_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    caffe_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    caffe_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    caffe_parser.add_argument('--input_pack_model_arrays', type=str, required=False, default='None',
                        help='Set input Pack model, specify name pack model like caffe(NCHW),comma-separated. All inputTersors will be NCHW if set "caffe" (default is NHWC)')
    caffe_parser.add_argument('--output_pack_model_arrays', type=str, required=False, default='None',
                        help='Set output Pack model, specify name pack model like caffe(NCHW),comma-separated. All outputTersors will be NCHW if set "caffe" (default is NHWC)')

    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
         caffe_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')


    #onnx model
    onnx_parser = subparsers.add_parser('onnx', help='onnx commands')
    onnx_parser.add_argument('--model_file', type=str, required=True,
                        help='Full filepath of tflite file containing the onnx model.')
    onnx_parser.add_argument('--input_arrays', type=str, required=True, default=None,
                        help='Names of the input arrays, comma-separated. (default None).')
    onnx_parser.add_argument('--input_shapes', type=str, required=True, default=None,
                            help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form N C H W (default None)')
    onnx_parser.add_argument('--output_arrays', type=str, required=True, default=None,
                        help='Names of the output arrays, comma-separated. (default None)')
    onnx_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    onnx_parser.add_argument('--output_file', type=str, required=True,
                        help='Full filepath of out Model path.')
    onnx_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    onnx_parser.add_argument('--input_pack_model', type=str, required=False, default="NCHW",
                        help='input tensor pack model,NCHW(caffe pytorch) or NHWC(tensorflow). default NCHW')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        onnx_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')

    return parser.parse_args()


def find_path(path, name):
    if path.split('/')[-1] == name:
        return path
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError('File `{}` not found in directory `{}`'.format(name, os.path.abspath(path)))


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


def main():
    args = arg_parse()
    platform = sys.argv[1]
    input_config = args.input_config
    output_file = args.output_file
    quant_file = args.quant_file
    debug = False
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        debug = args.debug
    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, 'Converted_Net_float.sim')
    elif (os.path.isdir(os.path.dirname(output_file)) or (os.path.dirname(output_file) == '')) :
        pass
    else:
        raise OSError('\033[31mCould not access path: {}\033[0m'.format(output_file))

    assert (os.path.exists(input_config))
    rawdata = False
    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read(input_config, encoding='utf-8')
    training_input_formats,input_formats = 'None','None'
    try:
        input_formats = conf.get('INPUT_CONFIG', 'input_formats')
    except:
        training_input_formats = conf.get('INPUT_CONFIG', 'training_input_formats')
    real_input_format = input_formats if input_formats!='None' else training_input_formats
    if real_input_format.replace(";","").split(',')[0] == 'RAWDATA_S16_NHWC':
        rawdata = True

    if platform == 'tensorflow_graphdef':
        graph_def_file = args.graph_def_file
        input_arrays_str = args.input_arrays
        output_arrays_str = args.output_arrays
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
        converter = tf.lite.TFLiteConverter.from_frozen_graph(
                graph_def_file, input_arrays, output_arrays, input_shapes)
    elif platform == 'tensorflow_savemodel':
        saved_model_dir = args.saved_model_dir
        input_arrays_str = args.input_arrays
        input_shapes_str = args.input_shapes
        output_arrays_str = args.output_arrays
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
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir,
                                                             input_arrays,
                                                             input_shapes,
                                                             output_arrays,
                                                             tag_set,
                                                             signature_key)
    elif platform == 'keras':
        model_file = args.model_file
        input_arrays_str = args.input_arrays
        input_shapes_str = args.input_shapes
        output_arrays_str = args.output_arrays
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

        converter = tf.lite.TFLiteConverter.from_keras_model_file(model_file,
                                                                  input_arrays,
                                                                  input_shapes,
                                                                  output_arrays,
                                                                  custom_objects)
    elif platform == 'tflite':
        model_file = args.model_file


    elif platform == 'caffe':
        model_file = args.model_file
        weight_file = args.weight_file
        input_arrays = args.input_arrays
        output_arrays = args.output_arrays
        input_pack_mode_arrays = args.input_pack_model_arrays
        output_pack_mode_arrays = args.output_pack_model_arrays
        converter = _convert_from_caffemodel(model_file ,
                                             weight_file,
                                             input_arrays,
                                             output_arrays,
                                             output_file,
                                             input_pack_mode_arrays,
                                             output_pack_mode_arrays,
                                             rawdata)



    elif platform == 'onnx':
        model_file = args.model_file
        input_arrays = args.input_arrays
        output_arrays = args.output_arrays
        input_shapes = args.input_shapes
        input_pack_model = args.input_pack_model
        converter = convert_from_onnx(model_file ,
                                             input_arrays,
                                             input_shapes,
                                             output_arrays,
                                             output_file,
                                             input_pack_model,
                                             rawdata)
    if platform == 'tflite':
        debug_output = model_file
    else:
        debug_output = output_file.strip().split('/')[-1]
        debug_output = output_file.replace(debug_output, 'Debug_' + debug_output)
        tflite_model = converter.convert()
        open(debug_output, "wb").write(tflite_model)

    #run sgs compiler ,convert debug tflite to sgs float tflite
    if 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        convert_tool_path = find_path(Project_path, 'convert_tool')
        compiler_config_path = find_path(Project_path, 'CompilerConfig.txt')
    elif 'TOP_DIR' in os.environ:
        Project_path = os.environ['TOP_DIR']
        sgs_tool_path = os.path.join(Project_path, '../SRC/Tool')
        Out_path = os.environ['OUT_DIR']
        convert_tool_path = os.path.join(Out_path, 'bin/convert_tool')
        compiler_config_path = find_path(sgs_tool_path, 'CompilerConfig.txt')
    else:
        raise OSError('Run source cfg_env.sh in top directory.')

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
        model_converter = calibrator_custom.converter(debug_output, input_config)
        new_compiler_config_path = calibrator_custom.utils.get_new_compiler_config()
        model_converter.convert(new_compiler_config_path, quant_info_list=quant_param, saved_path=output_file)
        os.remove(new_compiler_config_path)
    else:
        if debug:
            convert_cmd = 'gdb --args {} --model {} --output {} --transform {} --input_config {}'.format(
                            convert_tool_path, debug_output, output_file, compiler_config_path, input_config)
        else:
            convert_cmd = '{} --model {} --output {} --transform {} --input_config {}'.format(
                            convert_tool_path, debug_output, output_file, compiler_config_path, input_config)
        if debug:
            print('\033[33m================Debug command================\033[0m\n' + convert_cmd + '\n\033[33m=============================================\033[0m')
            os.system(convert_cmd)
        else:
            if os.path.exists(output_file):
                os.remove(output_file)
            proc = subprocess.Popen(convert_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = proc.communicate()
            if not os.path.exists(output_file):
                stdout = convert_to_unicode(stdout)
                stderr = convert_to_unicode(stderr)
                raise ConverterError("Convert failed. See console for info.\n%s\n%s\n" % (stdout, stderr))
    print('\nFloat model at: %s\n' % (output_file))

if __name__ == '__main__':
    main()
