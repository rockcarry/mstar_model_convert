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
import hashlib
import os
import copy
import shutil
import pdb
import six


from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter as cvt
from mace.python.tools.converter_tool import transformer
from mace.python.tools.convert_util import mace_check


FLAGS = None
DUMP_MACE = False


def convert_from_onnx(model_file,
                                       input_nodes,
                                       input_shapes,
                                       output_nodes,
                                       output_file,
                                       input_pack_model,
                                       rawdata=False):

    if not os.path.exists(model_file):
        raise NameError('Input graph file {} does not exist!'.format(model_file))


    option = cvt.ConverterOption()

    input_node_names = input_nodes.split(',')
    for i in six.moves.range(len(input_node_names)):
        input_node = cvt.NodeInfo()
        input_node.name = input_node_names[i]
        option.add_input_node(input_node)
    output_node_names = output_nodes.split(',')
    output_dir = output_file
    if os.path.isdir(output_dir):
        output_dir = os.path.join(output_dir, 'Converted_Net_float.sim')
    elif (os.path.isdir(os.path.dirname(output_dir)) or (os.path.dirname(output_dir) == '')) :
        pass
    else:
        raise OSError('\033[31mCould not access path: {}\033[0m'.format(output_dir))
    for i in six.moves.range(len(output_node_names)):
        output_node = cvt.NodeInfo()
        output_node.name = output_node_names[i]
        option.add_output_node(output_node)

    option.build()
    from mace.python.tools.converter_tool import onnx_converter
    converter = onnx_converter.OnnxConverter(option,
                                               input_shapes,
                                               model_file,
                                               rawdata=False)

    #output_graph_def = converter.run()
    output_graph_def,input_node_shapes_array = converter.run()
    from mace.python.tools.onnx import SGSModel_converter_onnx
    if DUMP_MACE:
      file_name = output_dir.strip("./").split(".")[0]
      f = open(file_name+'_mace.txt', 'w')
      f.write(str(output_graph_def))
      f.close()
    convertSGS = SGSModel_converter_onnx.ConvertSGSModel(output_graph_def, input_node_names, input_node_shapes_array, output_node_names, output_dir, input_pack_model)
    return convertSGS
