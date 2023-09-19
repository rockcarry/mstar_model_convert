# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import numpy as np
import argparse
import json
import pickle
from calibrator_custom import utils
from calibrator_custom import printf


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


def arg_parse():
    parser = argparse.ArgumentParser(description='Calibrator Tool')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    if 'TOP_DIR' in os.environ:
        parser.add_argument('-p', '--phase', type=str, default='Fixed',
                            choices=['Fixed', 'Fixed_without_ipu_ctrl'],
                            help='Indicate calibration phase.')
    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        parser.add_argument('--work_mode', type=str, default=None,
                            choices=['single_core', 'multi_core'],
                            help='Indicate calibrator work_mode.')
    parser.add_argument('--show_log', default=False, action='store_true',
                        help='Show log on screen.')
    parser.add_argument('--quant_file', type=str, default=None,
                        help='Import quantization file to modify model quantizaiton parameters. (JSON or Pickle)')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output path for fixed model.')

    return parser.parse_args()


def main():
    args = arg_parse()
    model_path = args.model
    input_config = args.input_config
    quant_file = args.quant_file
    output = args.output
    log = args.show_log
    work_mode = None
    if calibrator_custom.utils.VERSION[:2] in ['S6'] and args.work_mode is not None:
        work_mode = args.work_mode

    if 'TOP_DIR' in os.environ:
        calibrator_custom.SIM_calibrator.TARGET_TYPE = args.phase

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))

    if not os.path.exists(input_config):
        raise FileNotFoundError('input_config.ini file not found.')

    net = Net(model_path, input_config, work_mode, log)
    printf(str(net))

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

    out_model = utils.get_out_model_name(model_path, output, calibrator_custom.SIM_calibrator.TARGET_TYPE)

    if quant_param is None:
        net.convert_fixed(fix_model=[out_model])
    else:
        net.convert(quant_param=quant_param, fix_model=[out_model])

    printf('\nFixed model at: %s\n' % (out_model))

if __name__ == '__main__':
    main()
