# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
import time
import sys
from multiprocessing import Process
from utils import misc
from calibrator_custom import utils
import calibrator_custom
import pdb

def arg_parse():
    parser = argparse.ArgumentParser(description='Compiler Tool')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output path for offline model.')
    parser.add_argument('--input_config', type=str, default=None,
                        help='Input config path.')
    if utils.get_sdk_version() not in ['1', 'Q_0']:
        parser.add_argument('-l', '--lfw', type=str, default=None, help='Firmware total file path.')
        parser.add_argument('-b', '--batch', default='1', required=False, type=str,
            help='Expected max batch size in normal batch mode,or specific batches, comma-separated for multi-batches')
        parser.add_argument('--batch_mode', type=str, required=False, default='n_buf',
                            choices=['n_buf', 'one_buf'],
                            help='Expected batch mode.')
    parser.add_argument('-c', '--category', type=str, default='Unknown',
                        choices=['Classification', 'Detection', 'Unknown'],
                        help='Indicate net category.')
    parser.add_argument('-t', '--tool', default=None, type=str, help='sgs_calibration path.')
    parser.add_argument('-k', '--pack_tool', default=None, type=str, help='Pack_tool path.')
    parser.add_argument('--inplace_input_buf', type=str, default='True', help='inplace_input_buf,only True or False')
    parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        parser.add_argument('--debug', default=False, action='store_true', help='Run Debug mode.')
        if utils.get_sdk_version() not in ['1', 'Q_0']:
            parser.add_argument('-r', '--enable_test', default=False, action='store_true', help='Enable Pressure Test.')
            parser.add_argument('-x', '--random_seed', default=0, type=int, help='Random seed.')
    return parser.parse_args()


def gen_array_c(sgs_image_file):
    image_dir = os.path.dirname(os.path.abspath(sgs_image_file))
    image_basename = os.path.basename(sgs_image_file)
    os.chdir(image_dir)
    xxd_cmd = 'xxd -c 16 -i < {} > {}'.format(image_basename, image_basename + '_array.c')
    os.system(xxd_cmd)
    if not os.path.exists(image_basename + '_array.c'):
        print(xxd_cmd, 'failed!')#, file=sys.stderr)
    else:
        sed_cmd0 = r'sed -i "1i\#include \"common.h\"\n\nSGS_U8 SGS_DATA_ALIGN au8Model\[\] = {" %s' % (image_basename + '_array.c')
        sed_cmd1 = r'echo "};" >> %s' % (image_basename + '_array.c')
        os.system(sed_cmd0)
        os.system(sed_cmd1)
        print('Generate:', sgs_image_file + '_array.c')


def sgs_calibration_offline(pack, image_list, label, model, category, tool_path, pack_tool, convert_tool, output, debug=False):
    offline_net_file = '{}.offline.sim'.format(model)
    offline_cmd_file = '{}.cmdentry'.format(offline_net_file)
    if not pack:
        if debug:
            offline_cmd = 'gdb --args {} -m {} -i {} -l {} -p convert_offline -c {}'.format(
                tool_path, model, image_list[0], label, category)
            print('\033[33m================Debug command================\033[0m\n' + offline_cmd + '\n\033[33m=============================================\033[0m')
        else:
            offline_cmd = '{} -m {} -i {} -l {} -p convert_offline -c {} >> {}_offline.log'.format(
                tool_path, model, image_list[0], label, category, model.split('/')[-1])
        ret = os.system(offline_cmd) // 256
        if not (os.path.exists(offline_net_file) and (os.path.exists(offline_cmd_file))):
            raise RuntimeError('Run Calibration Offline failed!\nUse command to debug: {}'.format(offline_cmd))
    else:
        sgs_image_file = '{}_sgsimg.img'.format(model)
        if output is not None:
            if os.path.isdir(output):
                sgs_image_file = os.path.join(output, os.path.basename(sgs_image_file))
            elif (os.path.isdir(os.path.dirname(output)) or (os.path.dirname(output) == '')):
                sgs_image_file = output
        pack_tool_path = misc.find_path(pack_tool, 'pack_tool')
        pack_cmd = '{} -m {} -c {} -o {} >> {}_offline.log'.format(
            pack_tool_path, offline_net_file, offline_cmd_file, sgs_image_file, model.split('/')[-1]
        )
        ret = os.system(pack_cmd) // 256
        if not (os.path.exists(sgs_image_file)):
            raise RuntimeError('Run Pack tool failed!\nUse command to debug: {}'.format(pack_cmd))
        os.remove(offline_cmd_file)
        os.remove(offline_net_file)
        print('\033[31mOffline model at: {}\033[0m'.format(sgs_image_file))
        if convert_tool:
            gen_array_p = Process(target=gen_array_c, args=(sgs_image_file,))
            gen_array_p.start()
            gen_array_p.join()
    return ret

def sgs_calibration_offline_S(model, convert_tool, output, lfw, batchSize= 1 , batchMode='n_buf',enable_test=False, random_seed=0,inplace_input_buf=True,show_log=False,debug_mode=False,input_config=None):
    from calibrator_custom import printf
    offline_net_file = './output/{}.offline.sim'.format(os.path.basename(model))
    offline_cmd_file = './output/{}.1.cmdentry'.format(os.path.basename(model))
    mmu_cfg_file = './output/{}.1.mmucfg'.format(os.path.basename(model))

    inplace_input_buf_flag = True
    inplace_input_format_flag = ''
    misc.renew_folder('output')
    convert_model = calibrator_custom.compiler(model,show_log=show_log,enablePressureTest=enable_test,randomSeed=random_seed,input_config=input_config)
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

    if not (os.path.exists(offline_net_file)):
        raise RuntimeError('Run Calibration Offline failed!')


    printf('\033[31mRun Offline OK. Cost time: {}.\033[0m'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

    printf('\033[31mRun Offline OK.\033[0m')
    printf('\033[31mStart to run pack tool...\033[0m')

    #pack
    sgs_image_file = '{}_sgsimg.img'.format(model)
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

    if len(swdisp) == 0:
        convert_model.PackTool(offline_net_file, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file)
    else:
        convert_model.PackTool(offline_net_file, cmd_file, mmu_file, cmdfile_num_list, mmufile_num_list, lfw, batchMode, sgs_image_file,swdisp=swdisp)

    if not (os.path.exists(sgs_image_file)):
        raise RuntimeError('Run Pack tool failed!\n')

    for filename in cmd_file:
        os.remove(filename)
    for filename in mmu_file:
        os.remove(filename)
    os.remove(offline_net_file)
    printf('\033[31mOffline model at: {}\033[0m'.format(sgs_image_file))
    printf('\033[31mRun Pack Tool OK.\033[0m')
    return 1

def SGS_compiler():
    main()

def main():
    args = arg_parse()
    model = args.model
    input_config = args.input_config
    category = 'Unknown'
    tool_path = args.tool
    pack_tool = args.pack_tool
    output = args.output
    if utils.get_sdk_version() not in ['1', 'Q_0']:
        lfw = args.lfw
        batchSize = args.batch
        batchMode = args.batch_mode
        if batchMode == 'n_buf' and not batchSize.isdigit():
            raise ValueError('n_buf batch_mode must be specific one max batch size.')
    try:
        debug = args.debug
    except AttributeError:
        debug = False
    try:
        enable_test = args.enable_test
        random_seed = args.random_seed
    except AttributeError:
        enable_test = False
        random_seed = 0

    img_list = []

    if not os.path.exists(model):
        raise FileNotFoundError('No such model: {}'.format(model))
    model = os.path.abspath(model)

    if 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        image_path = os.path.join(Project_path, 'Scripts/calibrator/fix2Sgs.bmp')
        img_list.append(image_path)
        move_log = misc.Move_Log(clean=True)
        convert_tool = False
        if utils.get_sdk_version() in ['1', 'Q_0']:
            if pack_tool is None:
                pack_tool = Project_path
            if tool_path is None:
                tool_path = misc.find_path(Project_path, 'sgs_compiler')
        if utils.get_sdk_version() not in ['1', 'Q_0']:
            if lfw is None:
                lfw = os.path.join(Project_path, 'bin/ipu_lfw.bin')
            else:
                if not os.path.exists(lfw):
                    raise FileNotFoundError('{} not found'.format(lfw))
    elif 'TOP_DIR' in os.environ:
        Project_path = os.environ['TOP_DIR']
        image_path = os.path.join(Project_path, '../Tool/Scripts/calibrator/fix2Sgs.bmp')
        img_list.append(image_path)
        move_log = misc.Move_Log(clean=False)
        convert_tool = True
        if utils.get_sdk_version() in ['1', 'Q_0']:
            if pack_tool is None:
                pack_tool = Project_path
            if tool_path is None:
                tool_path = misc.find_path(Project_path, 'sgs_calibration')
        if utils.get_sdk_version() not in ['1', 'Q_0']:
            if lfw is None:
                lfw = misc.find_path(Project_path, 'ipu_lfw.bin')
            else:
                if not os.path.exists(lfw):
                    raise FileNotFoundError('{} not found'.format(lfw))
    else:
        raise OSError('Run source cfg_env.sh in top directory.')

    label_file = misc.Fake_Label(os.path.basename(model).split('.')[0])
    label = label_file.label_name

    if args.inplace_input_buf == 'True':
        inplace_input_buf = True
    else:
        inplace_input_buf = False

    if utils.get_sdk_version() not in ['1', 'Q_0']:
        from calibrator_custom import printf
        print('\033[31mStart to run convert offline network...\033[0m')
        if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
            ret = sgs_calibration_offline_S(model, convert_tool, output, lfw, batchSize, batchMode, args.enable_test, args.random_seed, inplace_input_buf, args.show_log,args.debug,input_config)
        else:
            ret = sgs_calibration_offline_S(model, convert_tool, output, lfw, batchSize, batchMode, False, 0, inplace_input_buf, args.show_log,input_config=input_config)

    else:
        print('\033[31mStart to run convert offline network...\033[0m')
        ret = sgs_calibration_offline(False, img_list, label, model, category, tool_path, pack_tool, convert_tool, output, debug)
        if (ret != 0):
            print('convert offline model return error code {}'.format(ret))
            return ret
        print('\033[31mRun Offline OK.\033[0m')
        print('\033[31mStart to run pack tool...\033[0m')
        ret = sgs_calibration_offline(True, img_list, label, model, category, tool_path, pack_tool, convert_tool, output, debug)
        print('\033[31mRun Pack Tool OK.\033[0m')
        if (ret != 0):
            print('pack model return error code {}'.format(ret))
            return ret


if __name__ == '__main__':
    sys.exit(main())
