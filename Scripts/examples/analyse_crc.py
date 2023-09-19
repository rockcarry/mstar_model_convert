# -*- coding: utf-8 -*-

import argparse
import sys
from calibrator_custom.performance_timeline_lib import GenerateCrcCheck

if __name__ == '__main__':

    parser=argparse.ArgumentParser(description='analyse crc log script')
    parser.add_argument('--f0c0',help='file0 core0 ipu log file path',type=str, required=False, default=None)
    parser.add_argument('--f0c1',help='file0 core1 ipu log file path',type=str, required=False, default=None)
    parser.add_argument('--f0cc',help='file0 corectrl ipu log file path',type=str,required=False, default=None)
    parser.add_argument('--f1c0',help='file1 core0 ipu log file path',type=str, required=False, default=None)
    parser.add_argument('--f1c1',help='file1 core1 ipu log file path',type=str, required=False, default=None)
    parser.add_argument('--f1cc',help='file1 corectrl ipu log file path',type=str,required=False, default=None)
    parser.add_argument('--ipu_version','-v',help='ipu chip version', choices=['36X', '93X', '37X', '938X'], type=str, default ='36X')
    args = parser.parse_args()
    if args.f0c0 is None and args.f0c1 is None and args.f0cc is None:
        raise ValueError('Need set file0 ipu log file path!!')
    if args.f1c0 is None and args.f1c1 is None and args.f1cc is None:
        raise ValueError('Need set file1 ipu log file path!!')

    gc = GenerateCrcCheck(args.f0c0, args.f0c1, args.f0cc, args.f1c0, args.f1c1, args.f1cc, args.ipu_version)
    gc.run()

