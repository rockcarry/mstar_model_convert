# -*- coding: utf-8 -*-
import argparse
from postprocess_method import *
import importlib
import os
import sys
import pdb

def arg_parse():
    parser = argparse.ArgumentParser(description='PostProcess Tool')
    parser.add_argument('-n', '--model_name', type=str, default='0', required=True,
                        help='Name of model to select image postprocess method')
    return parser.parse_args()

def main():
    args = arg_parse()
    model_name = args.model_name
    if os.path.exists(model_name) and model_name.split('.')[-1] == 'py':
        sys.path.append(os.path.dirname(model_name))
        postprocess_func = importlib.import_module(os.path.basename(model_name).split('.')[0])
        postprocess_func.model_postprocess()
    else:
        eval(model_name).model_postprocess()

if __name__ == '__main__':
    main()
