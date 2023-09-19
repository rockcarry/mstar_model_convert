import calibrator_custom
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Show SDK Info')
    parser.add_argument('--version', action='store_true', required=True,
                        help='Print SDK version info.')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    if args.version:
        print(calibrator_custom.__version__)

