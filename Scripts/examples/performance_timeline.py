# -*- coding: utf-8 -*-

import argparse
import sys
from calibrator_custom.performance_timeline_lib import GenerateTimeline

if __name__ == '__main__':

    parser=argparse.ArgumentParser(description='analyse performance timeline')
    parser.add_argument('--core0','-c0',help='core0 ipu log file path',type=str, required=False, default=None)
    parser.add_argument('--core1','-c1',help='core1 ipu log file path',type=str, required=False, default=None)
    parser.add_argument('--corectrl','-cc',help='corectrl ipu log file path',type=str,required=False, default=None)
    parser.add_argument('--frequency','-f',help='ipu clk freq',type=int, required=False, default=800)
    parser.add_argument('--ipu_version','-v',help='ipu chip version', choices=['36X', '93X', '37X', '938X'], type=str, default ='36X')
    args = parser.parse_args()
    if args.core0 is None and args.core1 is None and args.corectrl is None:
        raise ValueError('Need set ipu log file path!!')

    gt = GenerateTimeline(args.core0, args.core1, args.corectrl, args.frequency, args.ipu_version)
    gt.run()

