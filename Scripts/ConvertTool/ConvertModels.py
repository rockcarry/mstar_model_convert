# -*- coding: utf-8 -*-

import shutil
import os
import argparse
import pdb
import sys
import subprocess
import tensorflow as tf
from caffe_convert_tool import convert_from_caffemodel as _convert_from_caffemodel
from onnx_convert_tool import convert_from_onnx
import configparser
import re

def arg_parse():
    parser = argparse.ArgumentParser(description='Convert model')
    parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
    return parser.parse_args()

def find_path(path, name):
    if path.split('/')[-1] == name:
        return path
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError('File `{}` not found in directory `{}`'.format(name, os.path.abspath(path)))

def write_preprocess(preprocesspy_name, resizeH, resizeW, resizeC, meanB, meanG, meanR, std, rgb):
    with open(preprocesspy_name,'w') as f:
        content_lines = 'import cv2\nimport numpy as np\n' \
                'def get_image(img_path, resizeH={}, resizeW={}, resizeC={}, norm=True, meanB={}, meanG={}, meanR={}, std="{}", rgb={}):\n' \
                '   img = cv2.imread(img_path, flags=-1)\n' \
                '   if img is None:\n' \
                '       raise FileNotFoundError("No such image {}".format(img_path))\n' \
                '   try:\n' \
                '       img_dim = img.shape[2]\n' \
                '   except IndexError:\n' \
                '       img_dim = 1\n' \
                '   if resizeC == 3:\n' \
                '       if img_dim == 4:\n' \
                '           img = img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)\n' \
                '       elif img_dim == 1:\n' \
                '           img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)\n' \
                '   elif resizeC == 1:\n' \
                '       if img_dim == 3:\n' \
                '           img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n' \
                '       elif img_dim == 4:\n' \
                '           img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)\n' \
                '   img_norm = cv2.resize(img, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)\n' \
                '   if norm and resizeC ==3:\n' \
                '       img_norm = (img_norm - [meanB, meanG, meanR]) / [float(i) for i in std.split(":")]\n' \
                '       img_norm = img_norm.astype("float32")\n' \
                '   elif norm and (resizeC == 1):\n' \
                '       img_norm = (img_norm - meanB) / [float(i) for i in std.split(":")]\n' \
                '       img_norm = np.expand_dims(img_norm, axis=2)\n' \
                '       dummy = np.zeros((resizeH,resizeW,2))\n' \
                '       img_norm = np.concatenate((img_norm,dummy),axis=2)\n' \
                '       img_norm = img_norm.astype("float32")\n' \
                '   if rgb:\n' \
                '       img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)\n' \
                '   img_norm = np.expand_dims(img_norm, axis=0)\n' \
                '   return img_norm\n' \
                'def image_preprocess(img_path, norm=True):\n' \
                '   return get_image(img_path, norm=norm)'.format(resizeH, resizeW, resizeC, meanB, meanG, meanR, std, rgb,{})
        f.write(content_lines)

def convert_to_unicode(output):
    if output is None:
        return u""
    if isinstance(output, bytes):
        try:
            return output.decode()
        except UnicodeDecodeError:
            pass
    return output

def do_cmd(output_file, cmd_str):
    if os.path.exists(output_file):
            os.remove(output_file)
    proc = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = proc.communicate()
    if not os.path.exists(output_file):
        stdout = convert_to_unicode(stdout)
        stderr = convert_to_unicode(stderr)
        raise ConverterError("Convert failed. See console for info.\n%s\n%s\n" % (stdout, stderr))

class ConverterError(Exception):
    pass

def do_splitString(stringName):
    if re.search(r';$', stringName, re.I):
        stringNameNew = stringName.replace(';', '')
        return stringNameNew.strip()
    else:
        return stringName.strip()


def main():
    args = arg_parse()
    input_config = args.input_config 
    assert (os.path.exists(input_config))

    conf = configparser.ConfigParser()
    conf.read(input_config, encoding='utf-8')
    platform = do_splitString(conf.get('PLATFORM', 'platform'))
    special_case = do_splitString(conf.get('SPECIAL_MODEL_CASE', 'special_case'))
    input_files = do_splitString(conf.get('INPUT_MODEL_NAME', 'model_files'))
    output_file = do_splitString(conf.get('OUTPUT_MODEL_NAME', 'output_file'))
    input_shapes = do_splitString(conf.get('INPUT_CONFIG', 'input_shapes'))
    input_arrays = do_splitString(conf.get('INPUT_CONFIG','inputs'))
    output_arrays = do_splitString(conf.get('OUTPUT_CONFIG','outputs'))
    shapes_array = [shape_str for shape_str in input_shapes.strip().split(',')]
    mean_red = do_splitString(conf.get('INPUT_CONFIG', 'mean_red'))
    mean_green = do_splitString(conf.get('INPUT_CONFIG', 'mean_green'))
    mean_blue = do_splitString(conf.get('INPUT_CONFIG', 'mean_blue'))
    std_value = do_splitString(conf.get('INPUT_CONFIG', 'std_value'))
    training_input_formats = do_splitString(conf.get('INPUT_CONFIG', 'training_input_formats'))

    debug = False

    if 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        convert_tool_path = find_path(Project_path, 'ConvertTool.py')
        calibrator_path = find_path(Project_path,'calibrator.py')
        compiler_path = find_path(Project_path,'compiler.py')
        compiler_config_path = find_path(Project_path, 'CompilerConfig.txt')
        postprocess_path = find_path(Project_path, 'postprocess.py')
        concat_net_path = find_path(Project_path, 'concat_net')
    elif 'TOP_DIR' in os.environ:
        Project_path = os.environ['TOP_DIR']
        sgs_tool_path = os.path.join(Project_path, '../SRC/Tool')
        sgs_convertTool_path = os.path.join(Project_path, '../SRC/Tool/Scripts/ConvertTool')
        sgs_calibrator_path = os.path.join(Project_path, '../SRC/Tool/Scripts/calibrator')
        sgs_postprocess_path = os.path.join(Project_path, '../SRC/Tool/Scripts/postprocess')
        convert_tool_path = find_path(sgs_convertTool_path,'ConvertTool.py')
        calibrator_path = find_path(sgs_calibrator_path,'calibrator.py')
        compiler_path = find_path(sgs_calibrator_path,'compiler.py')
        compiler_config_path = find_path(sgs_tool_path, 'CompilerConfig.txt')
        postprocess_path = find_path(sgs_postprocess_path, 'postprocess.py')
        concat_net_path = find_path(Project_path, 'concat_net')
        debug = args.debug
        ipu = do_splitString(conf.get('PLATFORM', 'ipu'))
        source_cmd = 'cd {};make distclean;source build/{}/cfg_cmodel_float.sh;make -j99;cd -'.format(Project_path,ipu)
        print('=============source_cmd==============\n',source_cmd)
        os.system(source_cmd)
        debug = args.debug
    else:
        raise OSError('Run source cfg_env.sh in top directory.')
    if os.path.isdir(output_file) or output_file == '':
        output_file = os.path.join(output_file, 'Converted_Net_float.sim')
    elif (os.path.isdir(os.path.dirname(output_file)) or (os.path.dirname(output_file) == '')):
        pass
    else:
        raise OSError('\033[31mCould not access path: {}\033[0m'.format(output_file))

    if platform == 'caffe':
        model_files = input_files.split(',')
        model_file = model_files[0] if model_files[0].endswith('.prototxt') else model_files[1]
        weight_file = model_files[0] if model_files[0].endswith('.caffemodel') else model_files[1]
    else:
        model_file = input_files
    input_arrays_all = [input_str for input_str in input_arrays.split(',')]
    input_arrays1 = []
    for i in range(len(input_arrays_all)):
        if 'sgs_subnet_lstm' not in input_arrays_all[i]:
            input_arrays1.append(input_arrays_all[i])

    output_arrays_all = [output_str for output_str in output_arrays.split(',')]
    output_arrays1 = []
    for i in range(len(output_arrays_all)):
        if 'h_n' not in output_arrays_all[i]:
            output_arrays1.append(output_arrays_all[i])

    input_shapes_str_list = input_shapes.split(':') if input_shapes is not None else None

    if input_arrays1 is not None and input_shapes_str_list is not None and len(input_arrays1) != len(input_shapes_str_list):
        raise NameError('input_arrays\'s lengh is not equal input_shaps.')
    if (input_shapes_str_list is not None):
        input_shapes1 = {
                input_arrays1[t_num]: [int(i) for i in input_shapes_str_list[t_num].split(',')] for t_num in range(len(input_arrays1))
        }
    else:
        input_shapes1 = None

    if platform == 'tensorflow_graphdef':
        convert_cmd = 'python3 {} tensorflow_graphdef --graph_def_file {} --input_arrays {} --input_shapes {} --output_arrays {} --input_config {} --output_file {} '.format(convert_tool_path,
                    model_file, input_arrays, input_shapes, output_arrays, input_config, output_file)
        os.system(convert_cmd)


    elif platform == 'tensorflow_savemodel':
        tag_set = do_splitString(conf.get('SPECIAL_PARAMS','tag_set'))
        signature_key = do_splitString(conf.get('SPECIAL_PARAMS','signature_key'))
        convert_cmd = 'python3 {} tensorflow_savemodel --saved_model_dir {} --input_config {} --output_file {} --tag_set {} --signature_key {} '.format(convert_tool_path,
                    model_file, input_config, output_file, tag_set, signature_key)
        os.system(convert_cmd)


    elif platform == 'keras':
        convert_cmd = 'python3 {} keras --model_file {} --input_config {} --output_file {}'.format(convert_tool_path,
                    model_file, input_config, output_file)
        os.system(convert_cmd)

    elif platform == 'tflite':
        convert_cmd = 'python3 {} tflite --model_file {} --input_config {} --output_file {}'.format(convert_tool_path,
                    model_file, input_config, output_file)
        os.system(convert_cmd)

    elif platform == 'caffe':
        input_pack_model_arrays = do_splitString(conf.get('SPECIAL_PARAMS','input_pack_model_arrays'))
        output_pack_model_arrays = do_splitString(conf.get('SPECIAL_PARAMS','output_pack_model_arrays'))
        if input_pack_model_arrays == '' or input_pack_model_arrays == 'None':
            input_pack_model_arrays = 'None'
        if output_pack_model_arrays == '' or output_pack_model_arrays == 'None':
            output_pack_model_arrays = 'None'
        convert_cmd = 'python3 {} caffe --model_file {} --weight_file {} --input_arrays {} --output_arrays {} --input_config {} --output_file {} --input_pack_model_arrays {} --output_pack_model_arrays {}'.format(convert_tool_path,
                    model_file, weight_file, input_arrays, output_arrays, input_config, output_file, input_pack_model_arrays, output_pack_model_arrays)
        os.system(convert_cmd)

    elif platform == 'onnx':
        convert_cmd = 'python3 {} onnx --model_file {} --input_arrays {} --input_shapes {} --output_arrays {} --input_config {} --output_file {}'.format(convert_tool_path,
                    model_file, input_arrays, input_shapes, output_arrays, input_config, output_file)
        os.system(convert_cmd)

    if special_case == 'lstm':
        #gen_LSTM_Model
        if not os.path.dirname(output_file) == '' and os.path.exists(os.path.dirname(output_file)):
            lstm_file_path = os.path.dirname(output_file)
            gen_lstm_model_cmd = 'cd {};python3 {} -n caffe_lstm_unroll; cd -'.format(lstm_file_path,postprocess_path)
            os.system(gen_lstm_model_cmd)
            print('=============gen_lstm_model_cmd==============\n',gen_lstm_model_cmd)
            if os.path.exists(os.path.join(lstm_file_path,'SGS_LSTM_sub1_unroll.sim')):
                lstm_sub_model = os.path.join(lstm_file_path,'SGS_LSTM_sub1_unroll.sim')
            else:
                raise FileNotFoundError('No such model: {}'.format(os.path.join(lstm_file_path,'SGS_LSTM_sub1_unroll.sim')))

        else:
            gen_lstm_model_cmd = 'python3 {} -n caffe_lstm_unroll'.format(postprocess_path)
            os.system(gen_lstm_model_cmd)
            print('=============gen_lstm_model_cmd==============\n',gen_lstm_model_cmd)
            if os.path.exists('./SGS_LSTM_sub1_unroll.sim'):
                lstm_sub_model = './SGS_LSTM_sub1_unroll.sim'
            else:
                raise FileNotFoundError('No such model: {}'.format('SGS_LSTM_sub1_unroll.sim'))


        #output_file
        if re.search(r'_float\.sim$', output_file, re.I):
            output_concat_file = output_file.replace('_float.sim', '_concat_float.sim')
        else:
            pass

        concatnet_cmd = '{} --mode concat --transform {} --input_config {} --model1 {} --model2 {} --output {}'.format(concat_net_path,compiler_config_path,input_config,output_file,lstm_sub_model,output_concat_file)
        print(concatnet_cmd)
        os.system(concatnet_cmd)
        print('\nFloat model at: %s\n' % (output_concat_file))
        output_file = output_concat_file
        #concat backbone and lstm --> lstm_float_concat.sim

    if not training_input_formats == '' and 'RAWDATA_S16_NHWC' in training_input_formats.split(',') :
        preprocesspy_name = do_splitString(conf.get('SPECIAL_PARAMS','preprocesspy_name'))
        if not os.path.exists(preprocesspy_name):
            raise FileNotFoundError('No such model: {}'.format(preprocesspy_name))
    elif not do_splitString(conf.get('SPECIAL_PARAMS','preprocesspy_name')) == '':
        preprocesspy_name = do_splitString(conf.get('SPECIAL_PARAMS','preprocesspy_name'))
        if not os.path.exists(preprocesspy_name):
            raise FileNotFoundError('No such model: {}'.format(preprocesspy_name))
    else:
        ##preprocess
        preprocess_arrays = []
        input_arrays_all = [input_str for input_str in input_arrays.split(',')]
        if len(input_arrays_all)>1 and 'RAWDATA_S16_NHWC' not in training_input_formats.split(','):
            for i in range(len(input_arrays_all)):
                R = mean_red.split(',')[i]
                G = mean_green.split(',')[i]
                B = mean_blue.split(',')[i]
                std = std_value.split(',')[i]
                key = input_arrays_all[i]
                input_shape = input_shapes1[key]
                if platform == 'tensorflow_graphdef' or platform == 'tflite' or platform == 'tensorflow_savemodel':
                    resizeH = input_shape[1]
                    resizeW = input_shape[2]
                    resizeC = input_shape[3]
                else:
                    resizeH = input_shape[2]
                    resizeW = input_shape[3]
                    resizeC = input_shape[1]
                preprocesspy_name = 'preprocess_{}.py'.format(i)
                if not os.path.dirname(output_file) == '' and os.path.exists(os.path.dirname(output_file)):
                    preprocesspy_name = os.path.join(os.path.dirname(output_file),preprocesspy_name)
                preprocess_arrays.append(preprocesspy_name)
                write_preprocess(preprocesspy_name, resizeH, resizeW, resizeC, B, G, R, std, False)
        else:
            if platform == 'tensorflow_graphdef' or platform == 'tflite' or platform == 'tensorflow_savemodel':
                resizeH = shapes_array[1]
                resizeW = shapes_array[2]
                resizeC = shapes_array[3]
            else:
                resizeH = shapes_array[2]
                resizeW = shapes_array[3]
                resizeC = shapes_array[1]
            if not os.path.dirname(output_file) == '' and os.path.exists(os.path.dirname(output_file)):
                preprocesspy_name = os.path.join(os.path.dirname(output_file),'preprocess.py')
            else:
                preprocesspy_name = 'preprocess.py'
            preprocess_arrays.append(preprocesspy_name)
            write_preprocess(preprocesspy_name, resizeH, resizeW, resizeC, mean_blue, mean_green, mean_red, std_value, False)
        preprocesspy_name = ','.join(preprocess_arrays)
    images_path = do_splitString(conf.get('CALIBRATOR', 'images_path'))
    category_name = do_splitString(conf.get('CALIBRATOR', 'category_name'))
    num_process = do_splitString(conf.get('CALIBRATOR', 'num_process')) if do_splitString(conf.get('CALIBRATOR', 'num_process')) else 10
    if 'SGS_IPU_DIR' in os.environ:
        input_calibrate_model_file = output_file
        if re.search(r'_float\.sim$', input_calibrate_model_file, re.I):
            out_model = input_calibrate_model_file.replace('_float.sim', '_fixed.sim')
        elif re.search(r'_float_concat\.sim$', input_calibrate_model_file, re.I):
            out_model = input_calibrate_model_file.replace('_concat_float.sim', '_fixed.sim')
        else:
            out_model = input_calibrate_model_file.replace('.sim', '_fixed.sim')
        quant_level = do_splitString(conf.get('CALIBRATOR', 'quant_level'))
        if not quant_level =='':
            calibrator_cmd = 'python3 {} -i {} -m {} -c {} -n {} --input_config {} --num_process {} --quant_level {} --output {} '.format(calibrator_path,images_path,input_calibrate_model_file,category_name,preprocesspy_name,input_config,num_process,quant_level,out_model)
        else:
            calibrator_cmd = 'python3 {} -i {} -m {} -c {} -n {} --input_config {} --num_process {} --output {} '.format(calibrator_path,images_path,input_calibrate_model_file,category_name,preprocesspy_name,input_config,num_process,out_model)
        print('=============calibrator_cmd==============\n',calibrator_cmd)
        os.system(calibrator_cmd)
        if not os.path.exists(out_model):
            raise FileNotFoundError('No such model: {}'.format(out_model))
        compiler_cmd = 'python3 {} -m {}'.format(compiler_path,out_model)
        print('=============compiler_cmd==============\n',compiler_cmd)
        os.system(compiler_cmd)
        if re.search(r'_float\.sim$', input_calibrate_model_file, re.I):
            out_model = input_calibrate_model_file.replace('_float.sim', '_fixed.sim_sgsimg.img')
        else:
            out_model = input_calibrate_model_file.replace('.sim', '_fixed.sim_sgsimg.img')

        if not os.path.exists(out_model):
            raise FileNotFoundError('No such model: {}'.format(out_model))

    elif 'TOP_DIR' in os.environ:
        phase1 = 'Statistics'
        input_calibrate_model_file1 = output_file
        calibrator_cmd1 = 'python3 {} -i {} -m {} --input_config {} -p {} -n {}  '.format(calibrator_path,images_path,input_calibrate_model_file1,input_config,phase1,preprocesspy_name)
        print('=============calibrator_cmd1==============\n',calibrator_cmd1)
        os.system(calibrator_cmd1)
        if re.search(r'_float_concat\.sim$', input_calibrate_model_file1, re.I):
            out_model = input_calibrate_model_file1.replace('_concat_float.sim', '_cmodel_float.sim')
        elif re.search(r'_float\.sim$', input_calibrate_model_file1, re.I):
            out_model = input_calibrate_model_file1.replace('_float.sim','_cmodel_float.sim')
        else:
            out_model = input_calibrate_model_file1.replace('.sim', '_cmodel_float.sim')
        if not os.path.exists(out_model):
            raise FileNotFoundError('No such model: {}'.format(out_model)) 
        phase2 = 'Fixed'
        input_calibrate_model_file2 = out_model
        calibrator_cmd2 = 'python3 {} -i {} -m {} --input_config {} -p {} -n {}  '.format(calibrator_path,images_path,input_calibrate_model_file2,input_config,phase2,preprocesspy_name)
        print('=============calibrator_cmd2==============\n',calibrator_cmd2)
        os.system(calibrator_cmd2)
        if re.search(r'_float\.sim$', input_calibrate_model_file1, re.I):
            out_model = input_calibrate_model_file1.replace('_float.sim', '_fixed.sim')
        else:
            out_model = input_calibrate_model_file1.replace('.sim', '_fixed.sim')

        if not os.path.exists(out_model):
            raise FileNotFoundError('No such model: {}'.format(out_model))

        #offline  
        source_path = 'cd {};make distclean;source build/{}/cfg_cmodel_fixed_record_offline.sh;make distclean;source build/{}/cfg_cmodel_fixed_record_offline.sh;make -j99;cd -'.format(Project_path,ipu,ipu)
        print('=============source_path==============\n',source_path)
        os.system(source_path)
        compiler_cmd = 'python3 {} -m {}'.format(compiler_path,out_model)
        print('=============compiler_cmd==============\n',compiler_cmd)
        os.system(compiler_cmd)
        if re.search(r'_float\.sim$', input_calibrate_model_file1, re.I):
            out_model = input_calibrate_model_file1.replace('_float.sim', '_fixed.sim_sgsimg.img')
        else:
            out_model = input_calibrate_model_file1.replace('.sim', '_fixed.sim_sgsimg.img')

        if not os.path.exists(out_model):
            raise FileNotFoundError('No such model: {}'.format(out_model))

if __name__ == '__main__':
    main()
