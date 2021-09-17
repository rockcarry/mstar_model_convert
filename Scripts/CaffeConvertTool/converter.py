# Copyright 2018 The MACE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import os
import six


FLAGS = None

def main(unused_args):
    if not (('SGS_IPU_DIR' in os.environ) or ('TOP_DIR' in os.environ)):
        raise OSError('\033[31mRun `source cfg_env.sh` in Tool dir.\033[0m')

    if not os.path.exists(FLAGS.model_file):
        six.print_("Input graph file '" +
                   FLAGS.model_file +
                   "' does not exist!", file=sys.stderr)
        sys.exit(1)

    if FLAGS.platform == 'caffe':
        if not os.path.exists(FLAGS.weight_file):
            six.print_("Input weight file '" + FLAGS.weight_file +
                       "' does not exist!", file=sys.stderr)
            sys.exit(1)

    now_path = sys.argv[0]
    file_name = now_path.strip().split('/')[-1]
    convert_toolpy = now_path.replace(file_name, '../ConvertTool/ConvertTool.py')
    convert_cmd = 'python3 {} caffe --model_file {} --weight_file {} --input_arrays {} --output_arrays {} --input_config {} --output_file {} --input_pack_model_arrays {} --output_pack_model_arrays {}'.format(convert_toolpy,
                    FLAGS.model_file, FLAGS.weight_file, FLAGS.input_node, FLAGS.output_node, FLAGS.input_config, FLAGS.output_dir, FLAGS.input_pack_model_arrays, FLAGS.output_pack_model_arrays)
    os.system(convert_cmd)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description='Model Convert Tool'
    )
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_file", required=True, type=str, default="", help="Caffe prototxt file to load.")
    parser.add_argument(
        "--weight_file", required=True, type=str, default="", help="Caffe data file to load.")
    parser.add_argument(
        "--platform", type=str, default="caffe", help="caffe")
    parser.add_argument(
        "--input_node", required=True, type=str, default="input_node", help="e.g., input_node")
    parser.add_argument(
        "--output_node", required=True, type=str, default="softmax", help="e.g., softmax")
    parser.add_argument(
        "--output_dir", type=str, default="./Converted_Net_float.sim", help="File to save the output graph to.")
    parser.add_argument(
        '--input_config', type=str, required=True, help='Input config path.')
    parser.add_argument(
        '--input_pack_model_arrays', type=str, required=False, default=None, help='Set input Pack model, specify name pack model like caffe(NCHW),comma-separated. All inputTersors will be NCHW if set "caffe" (default is NHWC)')
    parser.add_argument(
        '--output_pack_model_arrays', type=str, required=False, default=None, help='Set output Pack model, specify name pack model like caffe(NCHW),comma-separated. All outputTersors will be NCHW if set "caffe" (default is NHWC)')
    return parser.parse_known_args()

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)
