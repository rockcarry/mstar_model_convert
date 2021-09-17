# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import numpy as np
import argparse
import json
import pickle
from calibrator_custom import utils


class Net(calibrator_custom.SIM_Calibrator):
    def __init__(self, model_path, input_config):
        super().__init__()
        self.model = calibrator_custom.calibrator(model_path, input_config)

    def forward(self, x):
        out_details = self.model.get_output_details()
        self.model.set_input(0, x)
        self.model.invoke()
        result_list = []
        for idx in range(len(out_details)):
            result = self.model.get_output(idx)
            result_list.append(result)
        return result_list


def arg_parse():
    parser = argparse.ArgumentParser(description='Calibrator Tool')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path.')
    parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output path for fixed model.')

    return parser.parse_args()


def main():
    args = arg_parse()
    model_path = args.model
    input_config = args.input_config
    output = args.output

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))

    if not os.path.exists(input_config):
        raise FileNotFoundError('input_config.ini file not found.')

    net = Net(model_path, input_config)
    print(net)

    out_model = utils.get_out_model_name(model_path, output)
    net.convert_fixed(fix_model=[out_model])


if __name__ == '__main__':
    main()
