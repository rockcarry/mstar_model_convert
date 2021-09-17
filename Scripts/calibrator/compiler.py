# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from utils import misc


def arg_parse():
    parser = argparse.ArgumentParser(description='Compiler Tool')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output path for offline model.')
    parser.add_argument('-c', '--category', type=str, default='Unknown',
                        choices=['Classification', 'Detection', 'Unknown'],
                        help='Indicate net category.')
    parser.add_argument('-t', '--tool', default='0', type=str, help='sgs_calibration path.')
    parser.add_argument('-k', '--pack_tool', default='0', type=str, help='Pack_tool path.')
    if ('TOP_DIR' in os.environ) and ('SGS_IPU_DIR' not in os.environ):
        parser.add_argument('--debug', default=False, action='store_true', help='Run gdb in Debug mode.')
    return parser.parse_args()


def sgs_calibration_offline(pack, image_list, label, model, category, tool_path, pack_tool, convert_tool, output, debug=False):
    offline_net_file = '{}.offline.sim'.format(model)
    offline_cmd_file = '{}.cmdentry'.format(offline_net_file)
    #fastrun_model_file = '{}.fastrunmodel'.format(offline_net_file)
    if not pack:
        if debug:
            offline_cmd = 'gdb --args {} -m {} -i {} -l {} -p convert_offline -c {}'.format(
                tool_path, model, image_list[0], label, category)
            print('\033[33m================Debug command================\033[0m\n' + offline_cmd + '\n\033[33m=============================================\033[0m')
        else:
            offline_cmd = '{} -m {} -i {} -l {} -p convert_offline -c {} >> {}_offline.log'.format(
                tool_path, model, image_list[0], label, category, model.split('/')[-1])
        os.system(offline_cmd)
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
        os.system(pack_cmd)
        if not (os.path.exists(sgs_image_file)):
            raise RuntimeError('Run Pack tool failed!\nUse command to debug: {}'.format(pack_cmd))
        os.remove(offline_cmd_file)
        os.remove(offline_net_file)
        #os.remove(fastrun_model_file)
        print('\033[31mOffline model at: {}\033[0m'.format(sgs_image_file))
        if convert_tool != '1':
            xxd_cmd = '{} -c 16 -i {} {}'.format(convert_tool, sgs_image_file, sgs_image_file + '_array.c')
            os.system(xxd_cmd)
            if not os.path.exists(sgs_image_file + '_array.c'):
                print(xxd_cmd, 'failed!', file=sys.err)
            else:
                with open(sgs_image_file + '_array.c', 'r') as f:
                    arrays = f.readlines()
                arrays[0] = '#include \"common.h\"\n\nSGS_U8 SGS_DATA_ALIGN au8Model[] = {\n'
                arrays[-1] = '\n'
                with open(sgs_image_file + '_array.c', 'w') as f:
                    f.writelines(arrays)
                print('Generate:', sgs_image_file + '_array.c')


def main():
    args = arg_parse()
    model = args.model
    category = 'Unknown'
    tool_path = args.tool
    pack_tool = args.pack_tool
    convert_tool = '0'
    output = args.output
    try:
        debug = args.debug
    except:
        debug = False

    img_list = []

    if not os.path.exists(model):
        raise FileNotFoundError('No such model: {}'.format(model))
    model = os.path.abspath(model)

    if 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        image_path = os.path.join(Project_path, 'Scripts/calibrator/fix2Sgs.bmp')
        img_list.append(image_path)
        move_log = misc.Move_Log(clean=True)
        if pack_tool == '0':
            pack_tool = Project_path
        if convert_tool == '0':
            convert_tool = '1'
        if tool_path == '0':
            tool_path = misc.find_path(Project_path, 'sgs_compiler')
    elif 'TOP_DIR' in os.environ:
        Project_path = os.environ['TOP_DIR']
        image_path = os.path.join(Project_path, '../SRC/Tool/Scripts/calibrator/fix2Sgs.bmp')
        img_list.append(image_path)
        move_log = misc.Move_Log(clean=False)
        if pack_tool == '0':
            pack_tool = Project_path
        if tool_path == '0':
            tool_path = misc.find_path(Project_path, 'sgs_calibration')
        if convert_tool == '0':
            convert_tool = 'xxd'
    else:
        raise OSError('Run source cfg_env.sh in top directory.')

    label_file = misc.Fake_Label('model_name')
    label = label_file.label_name
    print('\033[31mStart to run convert offline network...\033[0m')
    sgs_calibration_offline(False, img_list, label, model, category, tool_path, pack_tool, convert_tool, output, debug)
    print('\033[31mRun Offline OK.\033[0m')
    print('\033[31mStart to run pack tool...\033[0m')
    sgs_calibration_offline(True, img_list, label, model, category, tool_path, pack_tool, convert_tool, output, debug)
    print('\033[31mRun Pack Tool OK.\033[0m')



if __name__ == '__main__':
    main()
