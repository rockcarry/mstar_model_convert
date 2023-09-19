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
import configparser
import cv2
import pdb
from calibrator_custom import printf
if 'IPU_TOOL' in os.environ:
    Project_path = os.environ['IPU_TOOL']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/calibrator"))
elif 'SGS_IPU_DIR' in os.environ:
    Project_path = os.environ['SGS_IPU_DIR']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/calibrator"))
else:
    raise OSError('Run `source cfg_env.sh` in top directory.')
from utils import misc
import time
from multiprocessing import Process
from third_party.crypto import vendor_crypto
from third_party.crypto.vendor_crypto import *
from io import BytesIO

from calibrator_custom.ipu_quantization_lib import get_quant_config
from calibrator_custom.ipu_quantization_lib import quantize_pipeline, run_mpq, auto_pipeline
from calibrator_custom.ipu_quantization_lib import run_qat

import torch
import torch.backends.cudnn as cudnn
import pickle

cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.append("..")
from torch_calibrator import set_quant_config




class Net(calibrator_custom.SIM_Calibrator):
    def __init__(self, model_path, input_config, core_mode, log):
        super().__init__()
        self.model = calibrator_custom.calibrator(model_path, input_config, work_mode=core_mode, show_log=log)

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


def get_function(CLASS, function_name):
    for name, value in CLASS.__dict__.items():
        if name == function_name:
            return True
    return False


def sgs_calibration_offline_S(export_only_offline, orgin_model_name, model, convert_tool, output, lfw, batchSize= 1 , batchMode='n_buf',enable_test=False, random_seed=0,inplace_input_buf=True, show_log=False,debug_mode=False,is_decrypt=False):
    from calibrator_custom import printf
    if not export_only_offline:
        offline_net_file = './output/{}.offline.sim'.format(os.path.basename(model))
    else:
        offline_net_file = './output/sgs_model.offline.sim'
    orgin_model_name = os.path.basename(orgin_model_name)
    model_name = orgin_model_name[0:orgin_model_name.rfind('.')]
    inplace_input_buf_flag = True
    inplace_input_format_flag = ''
    misc.renew_folder('output')
    convert_model = calibrator_custom.compiler(model,show_log=show_log,enablePressureTest=enable_test,randomSeed=random_seed)
    input_list = convert_model.get_input_extern_details()
    for input_info in input_list:
        if input_info['dtype']==np.int16 and input_info['extflag'] & 1: # 1 represents E_TF_TENSOR_EXT_FLAG_FP32_2_S16
            inplace_input_buf_flag = inplace_input_buf
            inplace_input_format_flag = 'RAWDATA_F32_NHWC'

    if inplace_input_format_flag == 'RAWDATA_F32_NHWC':
        if batchMode == 'n_buf':
            if inplace_input_buf_flag == False:
                raise RuntimeError('inplace_input_buf only support when onebuffer.')
    else:
        if inplace_input_buf_flag == False:
            raise RuntimeError('inplace_input_buf only support when RAWDATA_FP32_NHWC.')
    start_time = time.time()
    #convert
    if not export_only_offline:
        if batchMode == 'n_buf':
            batchSize = int(float(batchSize))
            convert_model.NBuffer(batchSize)
        else:
            batchSize = batchSize.split(",")
            batchSize_1 = []
            for batch in batchSize:
                batch = int(float(batch))
                batchSize_1.append(batch)
            convert_model.OneBuffer(batchSize_1,inplace_input_buf_flag,debug_mode)
    else:
        if batchMode == 'n_buf':
            batchSize = int(float(batchSize))
            offline_sim_buf = convert_model.NBuffer(batchSize)
        else:
            batchSize = batchSize.split(",")
            batchSize_1 = []
            for batch in batchSize:
                batch = int(float(batch))
                batchSize_1.append(batch)
            offline_sim_buf = convert_model.OneBuffer(batchSize_1,inplace_input_buf_flag,debug_mode)

    printf('Run Offline OK. Cost time: {}.'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

    printf('Run Offline OK.')
    printf('Start to run pack tool...')

    #pack
    sgs_image_file = '{}_sgsimg.img'.format(model_name)
    if output is not None:
        if os.path.isdir(output):
            sgs_image_file = os.path.join(output, os.path.basename(sgs_image_file))
        elif (os.path.isdir(os.path.dirname(output)) or (os.path.dirname(output) == '')):
            sgs_image_file = output

    cmd_file = []
    mmu_file = []
    for maindir, subdir, file_name_list in os.walk("./output"):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if os.path.basename(apath).split('.')[-1].lower() == 'cmdentry':
                cmd_file.append(apath)
            elif os.path.basename(apath).split('.')[-1].lower() == 'mmucfg':
                mmu_file.append(apath)

    if len(cmd_file) == 0:
        raise FileNotFoundError('No cmdentry found in {}'.format(os.path.dirname(model)))

    if len(mmu_file) == 0:
        raise FileNotFoundError('No mmucfg found in {}'.format(os.path.dirname(model)))

    cmd_file.sort(key=lambda x: int(x.split('.')[-2]))
    mmu_file.sort(key=lambda x: int(x.split('.')[-2]))
    cmdfile_num_list = []
    mmufile_num_list = []

    for filename in cmd_file:
        cmdfile_list = []
        cmdfile_list = filename.split(".")
        index = len(cmdfile_list)-2
        cmdfile_num_list.append(int(float(cmdfile_list[index])))

    for filename in mmu_file:
        mmufile_list = []
        mmufile_list = filename.split(".")
        index = len(mmufile_list)-2
        mmufile_num_list.append(int(float(mmufile_list[index])))

    swdisp_data_path = './output/swdispFuncZoneInfo.bin'
    swdisp = ''
    if os.path.exists(swdisp_data_path):
        swdisp = swdisp_data_path
    if not is_decrypt:
        if not export_only_offline:
            if len(swdisp) == 0:
                convert_model.PackTool(offline_net_file, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,is_decrypt=is_decrypt)
            else:
                convert_model.PackTool(offline_net_file, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,swdisp=swdisp,is_decrypt=is_decrypt)
        else:
            if len(swdisp) == 0:
                convert_model.PackTool(offline_sim_buf, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,is_decrypt=is_decrypt)
            else:
                convert_model.PackTool(offline_sim_buf, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,swdisp=swdisp,is_decrypt=is_decrypt)

    else:

        if not export_only_offline:
            if len(swdisp) == 0:
                img_file = convert_model.PackTool(offline_net_file, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,is_decrypt=is_decrypt)
            else:
                img_file = convert_model.PackTool(offline_net_file, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,swdisp=swdisp,is_decrypt=is_decrypt)
        else:
            if len(swdisp) == 0:
                img_file = convert_model.PackTool(offline_sim_buf, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,is_decrypt=is_decrypt)
            else:
                img_file = convert_model.PackTool(offline_sim_buf, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,swdisp=swdisp,is_decrypt=is_decrypt)
        img_file = BytesIO(img_file)
        is_encrypt = get_function(vendor_crypto,'encrypt')
        if is_encrypt:
            img_file = encrypt(img_file)
        with open(sgs_image_file, 'wb') as f:
            f.write(img_file.read())
            f.close()
    if not (os.path.exists(sgs_image_file)):
        raise RuntimeError('Run Pack tool failed!\n')

    for filename in cmd_file:
        os.remove(filename)
    for filename in mmu_file:
        os.remove(filename)
    if not export_only_offline:
        os.remove(offline_net_file)
        printf('Offline model at: {}'.format(sgs_image_file))
    printf('Run Pack Tool OK.')
    return 1


def arg_parse():
    parser = argparse.ArgumentParser(description='Convert Tool')

    subparsers = parser.add_subparsers(help='platform info')

    #tensorflow graphdef
    graphdef_parser = subparsers.add_parser('tensorflow_graphdef', help='tensorflow graphdef commands')
    graphdef_parser.add_argument('--model_file', type=str, required=True,
                       help='Full filepath of file containing frozen GraphDef')
    graphdef_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    graphdef_parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    graphdef_parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    graphdef_parser.add_argument('--input_shapes', type=str, required=False, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form batch size, input array height, input array width, input array depth. (default None)')
    graphdef_parser.add_argument('--output_file', type=str, required=False,
                        help='Full filepath of out Model path.')
    graphdef_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    graphdef_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    graphdef_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    graphdef_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')

    graphdef_parser.add_argument('--export_models', default=False, action='store_true',
                            help='export models of debug/float/cmodel_float/fix')

    #torch calibrator
    graphdef_parser.add_argument('--quant_config', type=str, default=None,
                        help='Quant config(yaml) path.')
    graphdef_parser.add_argument('-q', '--q_mode', type=str, default=None,
                        help='Set Quantization mode')
    graphdef_parser.add_argument('--q_param', type=str, default=None,
                        help='Set param for specific q_mode')
    graphdef_parser.add_argument('--cal_batchsize', type=int, default=100)

    # for quant aware training
    graphdef_parser.add_argument('--resume', type=int, default=0)
    graphdef_parser.add_argument('--torch_q_param', type=str, default=None,
                        help='(QAT) torch_params.pkl file for finetune. ')
    graphdef_parser.add_argument('--multi_gpu', type=int, default=0, help='(QAT) multi_gpu or not')
    graphdef_parser.add_argument('--visible_gpu', type=str, default="0,1,2,3",
                        help='(QAT) Select which gpu can be seen')
    graphdef_parser.add_argument('--local_rank', type=int, default=-1, help='(QAT) processing rank for distributed training')

    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        graphdef_parser.add_argument('--work_mode', type=str, default=None,
                                choices=['single_core', 'multi_core'],
                                help='Indicate calibrator work_mode.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        graphdef_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
        if utils.VERSION[0] == 'S':
            graphdef_parser.add_argument('-r', '--enable_test', default=False, action='store_true', help='Enable Pressure Test.')
            graphdef_parser.add_argument('-x', '--random_seed', default=0, type=int, help='Random seed.')
    if 'IPU_TOOL' in os.environ:
        graphdef_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #tensorflow save_model
    savemodel_parser = subparsers.add_parser('tensorflow_savemodel', help='tensorflow save_model commands')
    savemodel_parser.add_argument('--model_file', type=str, required=True,
                       help='SavedModel directory to convert')
    savemodel_parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    savemodel_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    savemodel_parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    savemodel_parser.add_argument('--input_shapes', type=str, required=False, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form batch size, input array height, input array width, input array depth. (default None)')
    savemodel_parser.add_argument('--tag_set', type=str, required=False, default=None,
                        help='Set of tags identifying the MetaGraphDef within the SavedModel to analyze. All tags in the tag set must be present. (default None)')
    savemodel_parser.add_argument('--signature_key', type=str, required=False, default=None,
                        help='Key identifying SignatureDef containing inputs and outputs.(default DEFAULT_SERVING_SIGNATURE_DEF_KEY)')
    savemodel_parser.add_argument('--output_file', type=str, required=False,
                        help='Full filepath of out Model path.')
    savemodel_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    savemodel_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    savemodel_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    savemodel_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')
    savemodel_parser.add_argument('--export_models', default=False, action='store_true',
                            help='export models of debug/float/cmodel_float/fix')
    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        savemodel_parser.add_argument('--work_mode', type=str, default=None,
                                choices=['single_core', 'multi_core'],
                                help='Indicate calibrator work_mode.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        savemodel_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
        if utils.VERSION[0] == 'S':
            savemodel_parser.add_argument('-r', '--enable_test', default=False, action='store_true', help='Enable Pressure Test.')
            savemodel_parser.add_argument('-x', '--random_seed', default=0, type=int, help='Random seed.')
    if 'IPU_TOOL' in os.environ:
        savemodel_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #keras h5
    keras_parser = subparsers.add_parser('keras', help='keras commands')
    keras_parser.add_argument('--model_file', type=str, required=True,
                       help='Full filepath of HDF5 file containing the tf.keras model.')
    keras_parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    keras_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    keras_parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    keras_parser.add_argument('--input_shapes', type=str, required=False, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form batch size, input array height, input array width, input array depth. (default None)')
    keras_parser.add_argument('--custom_objects', type=str, required=False, default=None,
                        help='Dict mapping names (strings) to custom classes or functions to be considered during model deserialization. (default None)')
    keras_parser.add_argument('--output_file', type=str, required=False,
                        help='Full filepath of out Model path.')
    keras_parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    keras_parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    keras_parser.add_argument('--postprocess', type=str, default=None, required=False,
                        help='Name of model to select image postprocess method')
    keras_parser.add_argument('--postprocess_input_config', type=str, default=None, required=False,
                        help='Postprocess input config path.')
    keras_parser.add_argument('--export_models', default=False, action='store_true',
                            help='export models of debug/float/cmodel_float/fix')
    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        keras_parser.add_argument('--work_mode', type=str, default=None,
                                choices=['single_core', 'multi_core'],
                                help='Indicate calibrator work_mode.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        keras_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
        if utils.VERSION[0] == 'S':
            keras_parser.add_argument('-r', '--enable_test', default=False, action='store_true', help='Enable Pressure Test.')
            keras_parser.add_argument('-x', '--random_seed', default=0, type=int, help='Random seed.')
    if 'IPU_TOOL' in os.environ:
        keras_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #tensorflow lite
    tflite_parser = subparsers.add_parser('tflite', help='tflite commands')
    tflite_parser.add_argument('--model_file', type=str, required=True,
                       help='Full filepath of tflite file containing the tflite model.')
    tflite_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    tflite_parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    tflite_parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    tflite_parser.add_argument('--output_file', type=str, required=False,
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

    tflite_parser.add_argument('--export_models', default=False, action='store_true',
                            help='export models of debug/float/cmodel_float/fix')
    tflite_parser.add_argument('--decrypt', default=False, action='store_true',
                            help='input model buf needed decrypt')

    #torch calibrator
    tflite_parser.add_argument('--quant_config', type=str, default=None,
                        help='Quant config(yaml) path.')
    tflite_parser.add_argument('-q', '--q_mode', type=str, default=None,
                        help='Set Quantization mode')
    tflite_parser.add_argument('--q_param', type=str, default=None,
                        help='Set param for specific q_mode')
    tflite_parser.add_argument('--cal_batchsize', type=int, default=100)

    # for quant aware training
    tflite_parser.add_argument('--resume', type=int, default=0)
    tflite_parser.add_argument('--torch_q_param', type=str, default=None,
                        help='(QAT) torch_params.pkl file for finetune. ')
    tflite_parser.add_argument('--multi_gpu', type=int, default=0, help='(QAT) multi_gpu or not')
    tflite_parser.add_argument('--visible_gpu', type=str, default="0,1,2,3",
                        help='(QAT) Select which gpu can be seen')
    tflite_parser.add_argument('--local_rank', type=int, default=-1, help='(QAT) processing rank for distributed training')

    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        tflite_parser.add_argument('--work_mode', type=str, default=None,
                                choices=['single_core', 'multi_core'],
                                help='Indicate calibrator work_mode.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        tflite_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
        if utils.VERSION[0] == 'S':
            tflite_parser.add_argument('-r', '--enable_test', default=False, action='store_true', help='Enable Pressure Test.')
            tflite_parser.add_argument('-x', '--random_seed', default=0, type=int, help='Random seed.')
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
    caffe_parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    caffe_parser.add_argument('--export_models', default=False, action='store_true',
                            help='export models of debug/float/cmodel_float/fix')
    caffe_parser.add_argument('--decrypt', default=False, action='store_true',
                            help='input model buf needed decrypt')
    caffe_parser.add_argument('--output_file', type=str, required=False,
                        help='Full filepath of out Model path.')
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
    caffe_parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    #torch calibrator
    caffe_parser.add_argument('--quant_config', type=str, default=None,
                        help='Quant config(yaml) path.')
    caffe_parser.add_argument('-q', '--q_mode', type=str, default=None,
                        help='Set Quantization mode')
    caffe_parser.add_argument('--q_param', type=str, default=None,
                        help='Set param for specific q_mode')
    caffe_parser.add_argument('--cal_batchsize', type=int, default=100)

    # for quant aware training
    caffe_parser.add_argument('--resume', type=int, default=0)
    caffe_parser.add_argument('--torch_q_param', type=str, default=None,
                        help='(QAT) torch_params.pkl file for finetune. ')
    caffe_parser.add_argument('--multi_gpu', type=int, default=0, help='(QAT) multi_gpu or not')
    caffe_parser.add_argument('--visible_gpu', type=str, default="0,1,2,3",
                        help='(QAT) Select which gpu can be seen')
    caffe_parser.add_argument('--local_rank', type=int, default=-1, help='(QAT) processing rank for distributed training')

    caffe_parser.add_argument('--inplace_input_buf', type=str, default='True', help='inplace_input_buf,only True or False')
    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        caffe_parser.add_argument('--work_mode', type=str, default=None,
                                choices=['single_core', 'multi_core'],
                                help='Indicate calibrator work_mode.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        caffe_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
        if utils.VERSION[0] == 'S':
            caffe_parser.add_argument('-r', '--enable_test', default=False, action='store_true', help='Enable Pressure Test.')
            caffe_parser.add_argument('-x', '--random_seed', default=0, type=int, help='Random seed.')
    if 'IPU_TOOL' in os.environ:
        caffe_parser.add_argument('--to_debug', default=False, action='store_true',
                            help='Convert to Debug.sim model')

    #onnx model
    onnx_parser = subparsers.add_parser('onnx', help='onnx commands')
    onnx_parser.add_argument('--model_file', type=str, required=True,
                        help='Full filepath of tflite file containing the onnx model.')
    onnx_parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    onnx_parser.add_argument('--input_shapes', type=str, required=True, default=None,
                        help='Shapes corresponding to --input_arrays, colon-separated. For many models each shape takes the form N C H W (default None)')
    onnx_parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method')
    onnx_parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    onnx_parser.add_argument('--output_file', type=str, required=False,
                        help='Full filepath of out Model path.')
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
    onnx_parser.add_argument('--export_models', default=False, action='store_true',
                            help='export models of debug/float/cmodel_float/fix')
    onnx_parser.add_argument('--decrypt', default=False, action='store_true',
                            help='input model buf needed decrypt')

    #torch calibrator
    onnx_parser.add_argument('--quant_config', type=str, default=None,
                        help='Quant config(yaml) path.')
    onnx_parser.add_argument('-q', '--q_mode', type=str, default=None,
                        help='Set Quantization mode')
    onnx_parser.add_argument('--q_param', type=str, default=None,
                        help='Set param for specific q_mode')
    onnx_parser.add_argument('--cal_batchsize', type=int, default=100)

    # for quant aware training
    onnx_parser.add_argument('--resume', type=int, default=0)
    onnx_parser.add_argument('--torch_q_param', type=str, default=None,
                        help='(QAT) torch_params.pkl file for finetune. ')
    onnx_parser.add_argument('--multi_gpu', type=int, default=0, help='(QAT) multi_gpu or not')
    onnx_parser.add_argument('--visible_gpu', type=str, default="0,1,2,3",
                        help='(QAT) Select which gpu can be seen')
    onnx_parser.add_argument('--local_rank', type=int, default=-1, help='(QAT) processing rank for distributed training')

    onnx_parser.add_argument('--inplace_input_buf', type=str, default='True', help='inplace_input_buf,only True or False')
    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        onnx_parser.add_argument('--work_mode', type=str, default=None,
                                choices=['single_core', 'multi_core'],
                                help='Indicate calibrator work_mode.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        onnx_parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
        if utils.VERSION[0] == 'S':
            onnx_parser.add_argument('-r', '--enable_test', default=False, action='store_true', help='Enable Pressure Test.')
            onnx_parser.add_argument('-x', '--random_seed', default=0, type=int, help='Random seed.')
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
    input_arrays_str = config['INPUT_CONFIG']['inputs']
    input_arrays_str = input_arrays_str.strip(string.punctuation)
    output_arrays_str = config['OUTPUT_CONFIG']['outputs']
    output_arrays_str = output_arrays_str.strip(string.punctuation)
    SGSModel_transform_onnx.INPUT_CONFIG_INI = input_config

    # 2.read'--input_pack_model_arrays' from ini
    add_output_pack_inINI = False
    if 'input_layouts' in config['INPUT_CONFIG']:
        list_input_pack_model_arrays = config['INPUT_CONFIG']['input_layouts'].strip(string.punctuation) # return list
        str_input_pack_model_arrays = ''.join(list_input_pack_model_arrays)
        input_pack_model_arrays = str_input_pack_model_arrays
    else:
        input_pack_model_arrays = 'None'

    # 3.read'--output_pack_model_arrays' from ini
    add_output_pack_inINI = False
    if 'output_formats' in config['OUTPUT_CONFIG']:
        print('required args: -- output_pack_model_arrays is True')
        list_output_pack_model_arrays = config['OUTPUT_CONFIG']['output_formats'].strip(string.punctuation) # return list
        str_output_pack_model_arrays = ''.join(list_output_pack_model_arrays)
        output_pack_model_arrays = str_output_pack_model_arrays
    elif 'output_layouts' in config['OUTPUT_CONFIG']:
        print('required args: -- output_pack_model_arrays is True')
        list_output_pack_model_arrays = config['OUTPUT_CONFIG']['output_layouts'].strip(string.punctuation) # return list
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

    # ## 6.read'--input_shapes' from ini
    # if platform == 'tensorflow_graphdef' or platform == 'tensorflow_savemodel' or platform == 'keras' or platform == 'onnx':
    #     input_shapes_str = config['INPUT_CONFIG']['input_shapes']
    #     input_shapes_str = input_shapes_str.strip(string.punctuation)
    # else:
    #     input_shapes_str = 'NONE'

    ## 11. read '--quant_file' from ini
    if 'quant_file'  in config['INPUT_CONFIG']:
        quant_file = config['INPUT_CONFIG']['quant_file']
        quant_file = quant_file.strip(';')
    else:
        quant_file = None
    ## 12. read '--num_process' from ini
    if 'num_process'  in config['INPUT_CONFIG']:
        num_process = config['INPUT_CONFIG']['num_process']
        num_process = num_process.strip(';')
        num_process = int(num_process)
    else:
        num_process = 10

    ## 13. read '--quant_level' from ini
    if 'quant_level'  in config['INPUT_CONFIG']:
        quant_level = config['INPUT_CONFIG']['quant_level']
        quant_level = quant_level.strip(';')
    else:
        quant_level = 'L5'

    ## 14. read '--phase' from ini
    if 'phase'  in config['INPUT_CONFIG']:
        phase = config['INPUT_CONFIG']['phase']
        phase = phase.strip(';')
    else:
        phase = 'Fixed'

    ## 15. read '--workmode' from ini
    if 'workmode' in config['INPUT_CONFIG']:
        workmode = config['INPUT_CONFIG']['workmode']
        workmode = workmode.strip(';')
    else:
        workmode = None

    ## 16. read '--batchmode' from ini
    if 'batchmode' in config['INPUT_CONFIG']:
        batchmode = config['INPUT_CONFIG']['batchmode']
        batchmode = batchmode.strip(';')
    else:
        batchmode = 'n_buf'

    ## 17. read '--batch' from ini
    if 'batch' in config['INPUT_CONFIG']:
        batch = config['INPUT_CONFIG']['batch']
        batch = batch.strip(';')
    else:
        batch = '1'

    return input_arrays_str, output_arrays_str, input_pack_model_arrays, output_pack_model_arrays, \
           quant_file,num_process,quant_level,phase,workmode,batchmode,batch,input_config

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
caffe_buffer_dic = {}


def SGS_Convert_One_Step(caffe_src_buf=None, caffe_weight_buf=None, onnx_buf=None, tflite_buf=None):
    caffe_buffer_dic = {}
    caffe_buffer_dic['caffe_src'] = caffe_src_buf
    caffe_buffer_dic['caffe_weight'] = caffe_weight_buf
    platform_buffer_map['caffe'] = caffe_buffer_dic
    platform_buffer_map['onnx'] = onnx_buf
    platform_buffer_map['tflite'] = tflite_buf
    main()

def main():
    args = arg_parse()
    platform = sys.argv[1]
    model_file = args.model_file
    input_config = args.input_config
    image_file = args.image
    quant_file = args.quant_file
    if platform == 'caffe' or platform == 'onnx' or platform == 'tflite':
        is_decrypt = args.decrypt
    else:
        is_decrypt = False
    if is_decrypt == True:
        if args.export_models:
            raise ValueError('please do not add --export_models when --decrypt  is given')
    fixedC2fixedWO = False
    lstm_flag = False
    output_file = 'sgs_float.sim'
    if args.export_models:
        if not args.output_file:
            raise ValueError('please input filename of float model with --output_file when --export_models is given')
        else:
            output_file = args.output_file
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

    input_arrays_str, output_arrays_str, input_pack_model_arrays, output_pack_model_arrays, quant_file\
    ,num_process,quant_level,phase,workmode,batchmode,batch,input_config = read_args_from_ini(input_config)

    if args.export_models == True:
        export_only_offline = False
    else:
        export_only_offline = True
    origin_model_name = model_file
    SGSModel_transform_onnx.INPUT_CONFIG_INI = input_config
    SGSModel_transform_onnx_S.INPUT_CONFIG_INI = input_config
    clear_directory(input_config)

    from calibrator_custom import printf
    printf('\033[32mStart to run convert float network...\033[0m')

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
#                graph_def_file = model_file
#        else:
#            graph_def_file = model_file
        graph_def_file = model_file
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
        model_file = tfliteModel_output_path
        if args.export_models == True:
            open(tfliteModel_output_path, "wb").write(tfliteModel)
            #tflite convert to mace
            converter = convert_from_tflite(model_file ,output_file)
        else:
            converter = convert_from_tflite(model_file ,output_file, model_buf=tfliteModel)

    elif platform == 'tensorflow_savemodel':
#        if 'tensorflow_savemodel' in platform_buffer_map:
#            if platform_buffer_map['tensorflow_savemodel'] != None:
#                saved_model_dir = platform_buffer_map['tensorflow_savemodel']
#            else:
#                saved_model_dir = model_file
#        else:
#            saved_model_dir = model_file

        saved_model_dir = model_file
        # input_arrays_str = args.input_arrays
        # output_arrays_str = args.output_arrays
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
        model_file = tfliteModel_output_path
        if args.export_models == True:
            open(tfliteModel_output_path, "wb").write(tfliteModel)
            #tflite convert to mace
            converter = convert_from_tflite(model_file ,output_file)
        else:
            converter = convert_from_tflite(model_file ,output_file, model_buf=tfliteModel)

    elif platform == 'keras':
        # input_arrays_str = args.input_arrays
        # output_arrays_str = args.output_arrays
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
        model_file = tfliteModel_output_path
        if args.export_models == True:
            open(tfliteModel_output_path, "wb").write(tfliteModel)
            #tflite convert to mace
            converter = convert_from_tflite(model_file ,output_file)
        else:
            converter = convert_from_tflite(model_file ,output_file, model_buf=tfliteModel)

    elif platform == 'tflite':
        tfliteModel_buf = None
        bFixed_model = args.fixed_model
        if 'tflite' in platform_buffer_map:
            if platform_buffer_map['tflite'] != None:
                tfliteModel_buf = platform_buffer_map['tflite']
            else:
                if is_decrypt == True:
                    with open(args.model_file, 'rb') as f:
                        tfliteModel_buf = BytesIO(f.read())
                else:
                    model_file = args.model_file
        else:
            if is_decrypt == True:
                with open(args.model_file, 'rb') as f:
                    tfliteModel_buf = BytesIO(f.read())
            else:
                model_file = args.model_file
        converter = convert_from_tflite(model_file, output_file, bFixed_model=bFixed_model, model_buf=tfliteModel_buf, is_decrypt=is_decrypt)
#        input_config = args.input_config
#        input_arrays = [input_str for input_str in input_arrays_str.split(',')]  if input_arrays_str is not None else None
#        output_arrays = [output_str for output_str in output_arrays_str.split(',')] if output_arrays_str is not None else None

    elif platform == 'caffe':
        if 'caffe' in platform_buffer_map:
            if platform_buffer_map['caffe']['caffe_src'] != None and platform_buffer_map['caffe']['caffe_weight'] != None:
                model_file = platform_buffer_map['caffe']['caffe_src']
                weight_file = platform_buffer_map['caffe']['caffe_weight']
            else:
                if is_decrypt == True:
                    with open(args.model_file, 'rb') as f:
                        model_file = BytesIO(f.read())
                    with open(args.weight_file, 'rb') as f:
                        weight_file = BytesIO(f.read())
                else:
                    model_file = args.model_file
                    weight_file = args.weight_file
        else:
            if is_decrypt == True:
                with open(args.model_file, 'rb') as f:
                    model_file = BytesIO(f.read())
                with open(args.weight_file, 'rb') as f:
                    weight_file = BytesIO(f.read())
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
                                             input_name_format_map,
                                             is_decrypt)

    elif platform == 'onnx':
        if 'onnx' in platform_buffer_map:
            if platform_buffer_map['onnx'] != None:
                model_file = platform_buffer_map['onnx']
            else:
                if is_decrypt == True:
                    with open(args.model_file, 'rb') as f:
                        model_file = BytesIO(f.read())
                else:
                    model_file = args.model_file
        else:
            if is_decrypt == True:
                with open(args.model_file, 'rb') as f:
                    model_file = BytesIO(f.read())
            else:
                model_file = args.model_file
        input_arrays = input_arrays_str
        output_arrays = output_arrays_str
        input_pack_model = input_pack_model_arrays
        output_pack_model = output_pack_model_arrays
        input_shapes_str = args.input_shapes
        fixedC2fixedWO = args.fixedC2fixedWO
        skip_simplify = args.skip_simplify
        if export_only_offline == True:
            saved_refine_onnx = False
        else:
            saved_refine_onnx = True
        converter = convert_from_onnx(model_file ,
                                             input_arrays,
                                             input_shapes_str,
                                             output_arrays,
                                             output_file,
                                             input_pack_model,
                                             output_pack_model,
                                             input_name_format_map,
                                             fixedC2fixedWO,
                                             skip_simplify,
                                             saved_refine_onnx,
                                             is_decrypt)

    if platform == 'debug':
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
           if export_only_offline == False:
                with open(debug_output, "wb") as f:
                    f.write(tflite_model)
           debug_model = tflite_model

        else:
            debug_output = args.model_file
    else:
        debug_output = os.path.basename(output_file)
        debug_output = output_file.replace(debug_output, 'Debug_' + debug_output)
        tflite_model = converter.convert()
        lstm_flag = True if converter._lstm_num > 0 else False
        if export_only_offline == False:
            with open(debug_output, "wb") as f:
                f.write(tflite_model)
        debug_model = tflite_model


    if lstm_flag:
        if 'IPU_TOOL' in os.environ:
            Project_path = os.environ['IPU_TOOL']
        elif 'SGS_IPU_DIR' in os.environ:
            Project_path = os.environ['SGS_IPU_DIR']
        else:
            raise OSError('Run `source cfg_env.sh` in top directory.')

        if export_only_offline == True:
            if platform == 'onnx':
                postprocess_file = os.path.join(Project_path,'Scripts/postprocess/sgs_chalk_postprocess_method/onnx_lstm_unroll_export_buf.py')
            else:
                postprocess_file = os.path.join(Project_path,'Scripts/postprocess/sgs_chalk_postprocess_method/caffe_lstm_unroll_export_buf.py')
        else:
            if platform == 'onnx':
                postprocess_file = os.path.join(Project_path,'Scripts/postprocess/sgs_chalk_postprocess_method/onnx_lstm_unroll.py')
            else:
                postprocess_file = os.path.join(Project_path,'Scripts/postprocess/sgs_chalk_postprocess_method/caffe_lstm_unroll.py')

        if os.path.exists(postprocess_file) and postprocess_file.split('.')[-1] == 'py':
            sys.path.append(os.path.dirname(postprocess_file))
            postprocess_func = importlib.import_module(os.path.basename(postprocess_file).split('.')[0])
            if export_only_offline == True:
                postfile_buf_list = postprocess_func.model_postprocess()
            else:
                postfile_list = postprocess_func.model_postprocess()
        else:
            #postfile_list = eval(postprocess_file).model_postprocess()
            raise ValueError('please input postprocess file with full path')

        model_file = debug_output
        if export_only_offline == True:
            converter = convert_from_tflite(model_file,
                                        output_file,
                                        mode='concat',
                                        model_buf = debug_model,
                                        subgraph_model_buf=postfile_buf_list
                                        )

        else:

            converter = convert_from_tflite(model_file,
                                        output_file,
                                        postfile_list,
                                        'concat',
                                        )

        debug_output = os.path.basename(output_file)
        debug_output = output_file.replace(debug_output, 'Concat_' + debug_output)
        concat_model = converter.convert()
        if export_only_offline == False:
            with open(debug_output, "wb") as f:
                f.write(concat_model)
        debug_model = concat_model

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

            postprocess_converter = tflite_converter.TfliteConverter(postfile_list,bFixed_model=False,model_buf=None)
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
                                       model_buf=None
                                       )

        debug_output = os.path.basename(output_file)
        debug_output = output_file.replace(debug_output, 'Concat_' + debug_output)
        concat_model = converter.convert()
        if export_only_offline == False:
            with open(debug_output, "wb") as f:
                f.write(concat_model)
        debug_model = concat_model


    if 'IPU_TOOL' in os.environ and args.to_debug:
        print('\nDebug model at: %s\n' % (debug_output))
        return

    if export_only_offline:
        debug_model_new = bytes(debug_model)
        model_converter = calibrator_custom.converter(debug_model_new, input_config, show_log=args.show_log)
    else:
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
        if not export_only_offline:
            model_converter.convert(compiler_config.Debug2FloatConfig(SkipInfectDtype=True), quant_info_list=quant_param, saved_path=output_file)
            print('\nFloat model at: %s\n' % (output_file))
        else:
            float_model = model_converter.convert(compiler_config.Debug2FloatConfig(SkipInfectDtype=True), quant_info_list=quant_param)
    else:
        if fixedC2fixedWO:
            if not export_only_offline:
                model_converter.convert(compiler_config.Float2FixedWOConfig(), saved_path=output_file, model_type='Fixed_without_ipu_ctrl')
                print('\nFixed model at: %s\n' % (output_file))
            else:
                float_model = model_converter.convert(compiler_config.Float2FixedWOConfig(), model_type='Fixed_without_ipu_ctrl')
        else:
            if not export_only_offline:
                model_converter.convert(compiler_config.Debug2FloatConfig(), saved_path=output_file, model_type='Float')
                print('\nFloat model at: %s\n' % (output_file))
            else:
                float_model = model_converter.convert(compiler_config.Debug2FloatConfig(), model_type='Float')


    #fix

    from calibrator_custom import printf
    printf('\n')
    printf('\033[32mStart to run convert fix network...\033[0m')
    image_path = image_file
    model_path = output_file
    model_name = args.preprocess
    num_subsets = num_process
    output = None
    log = args.show_log
    work_mode = None

    if calibrator_custom.utils.VERSION[:2] in ['S6'] and args.work_mode is not None:
        work_mode = args.work_mode

    if 'SGS_IPU_DIR' in os.environ:
        move_log = misc.Move_Log(clean=True)
    elif 'TOP_DIR' in os.environ:
        move_log = misc.Move_Log(clean=False)
        calibrator_custom.SIM_calibrator.TARGET_TYPE = phase
    else:
        raise OSError('\033[31mRun source cfg_env.sh in top directory.\033[0m')

    if export_only_offline:
        net = Net(float_model, input_config, work_mode, log)
    else:
        net = Net(model_path, input_config, work_mode, log)
    printf(str(net))

    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        if os.path.isdir(image_path):
            dir_name = None
            base_name = image_path
        elif os.path.isfile(image_path):
            dir_name = os.path.abspath(os.path.dirname(image_path))
            base_name = image_path
        else:
            dir_name = None
            base_name = image_path

    if args.q_mode is None:
        preprocess_funcs = [utils.image_preprocess_func(n) for n in model_name.split(',')]
        if os.path.isdir(base_name):
            image_list = utils.all_path(base_name)
            img_gen = utils.image_generator(image_list, preprocess_funcs)
        elif os.path.basename(base_name).split('.')[-1].lower() in utils.image_suffix:
            img_gen = utils.image_generator([base_name], preprocess_funcs)
        else:
            with open(base_name, 'r') as f:
                multi_images = f.readlines()
            if dir_name is None:
                multi_images = [images.strip().split(',') for images in multi_images]
            else:
                multi_images = [[os.path.join(dir_name, i) for i in images.strip().split(',')] for images in multi_images]
            img_gen = utils.image_generator(multi_images, preprocess_funcs)

        quant_param = None
        if quant_file is not None:
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

        if not export_only_offline:
            out_model = utils.get_out_model_name(model_path, output, calibrator_custom.SIM_calibrator.TARGET_TYPE)
            net.convert(img_gen, num_process=num_subsets, quant_level=quant_level, quant_param=quant_param,
            fix_model=[out_model])
            printf('\nFixed model at: %s\n' % (out_model))
        else:
            fix_model = net.convert(img_gen, num_process=num_subsets, quant_level=quant_level, quant_param=quant_param)
    else:
        quant_cfg = get_quant_config(
                    model_path,
                    quant_config=args.quant_config,
                    input_config=input_config,
                    python_file=model_name,
                    calset_dir=image_path,
                    image_list=None,
                    cal_batchsize=args.cal_batchsize
                )
        q_mode = args.q_mode.upper()
        args.model_file = model_path
        if q_mode != 'Q32':
            quant_cfg = set_quant_config(q_mode,quant_cfg,net,args)
        else:
            quant_cfg = set_quant_config('Q23',quant_cfg,net,args)
            printf('\nQ23 run suceess\n' )
            args.torch_q_param = quant_cfg.torch_params_file
            quant_cfg = set_quant_config(q_mode,quant_cfg,net,args)

        if args.local_rank in [0, -1]:
            out_model = utils.get_out_model_name(model_path, output, calibrator_custom.SIM_calibrator.TARGET_TYPE)
            # print('\033[92mFeed sim_params\033[0m')
            if quant_cfg.sim_params_file is not None:
                sim_params = pickle.load(open(quant_cfg.sim_params_file, 'rb'))
            else:
                sim_params=None
            if not export_only_offline:
                if quant_cfg.onnx_file.endswith('_float.sim') and sim_params is not None:
                    net.convert(quant_param=sim_params, fix_model=[out_model])
                    printf('\nFixed model at: %s\n' % (out_model))
                else:
                    raise ValueError('torch calibrator error.')
            else:
                if sim_params is not None:
                    fix_model = net.convert(quant_param=sim_params)

    printf('\nFixed network convert suceess\n' )

    #offline
    if not export_only_offline:
        model = out_model
    else:
        model = fix_model[0]
    batchSize = batch
    batchMode = batchmode
    img_list = []
    lfw = None
    pack_tool = None
    tool_path = None
    if 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        image_path = os.path.join(Project_path, 'Scripts/calibrator/fix2Sgs.bmp')
        img_list.append(image_path)
        move_log = misc.Move_Log(clean=True)
        convert_tool = False
        if utils.get_sdk_version() not in ['1', 'Q_0']:
            if lfw is None:
                lfw = os.path.join(Project_path, 'bin/ipu_lfw.bin')
    elif 'TOP_DIR' in os.environ:
        Project_path = os.environ['TOP_DIR']
        image_path = os.path.join(Project_path, '../Tool/Scripts/calibrator/fix2Sgs.bmp')
        img_list.append(image_path)
        move_log = misc.Move_Log(clean=False)
        convert_tool = True
        if utils.get_sdk_version() not in ['1', 'Q_0']:
            if lfw is None:
                lfw = misc.find_path(Project_path, 'ipu_lfw.bin')
    else:
        raise OSError('Run source cfg_env.sh in top directory.')

    if platform == 'onnx' or platform == 'caffe':
        if args.inplace_input_buf == 'True':
            inplace_input_buf = True
        else:
            inplace_input_buf = False
    else:
        inplace_input_buf = True
    if utils.VERSION[0] == 'S':
        from calibrator_custom import printf
        printf('\033[32mStart to run convert offline network...\033[0m')
        if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
            ret = sgs_calibration_offline_S(export_only_offline,origin_model_name, model, convert_tool, None, lfw, batchSize, batchMode, args.enable_test, args.random_seed, inplace_input_buf, args.show_log,is_decrypt=is_decrypt)
        else:
            ret = sgs_calibration_offline_S(export_only_offline,origin_model_name, model, convert_tool, None, lfw, batchSize, batchMode, False, 0, inplace_input_buf, args.show_log,is_decrypt=is_decrypt)

if __name__ == '__main__':
    main()
