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


def convert_from_tflite(model_file,output_file,postprocess_file_list=None,mode=None,bFixed_model=False,convert_kb=None,model_buf=None,subgraph_model_buf=None,is_decrypt=None):
    postmodel_list = []
    if model_buf is None:
        if not os.path.exists(model_file):
            raise NameError('Input graph file {} does not exist!'.format(model_file))
    output_dir = output_file
    if os.path.isdir(output_dir):
        output_dir = os.path.join(output_dir, 'Converted_Net_float.sim')
    elif (os.path.isdir(os.path.dirname(output_dir)) or (os.path.dirname(output_dir) == '')) :
        pass
    else:
        raise OSError('\033[31mCould not access path: {}\033[0m'.format(output_dir))

    from mace.python.tools.converter_tool import tflite_converter
    converter = tflite_converter.TfliteConverter(model_file, bFixed_model,model_buf,is_decrypt)
    tflite2mace_graph_def_model, list_input_name, list_output_name = converter.run()

    from mace.python.tools import SGSModel_converter_from_Mace

    if mode is not None:
        if postprocess_file_list is not None:
            postprocess_file_num = len(postprocess_file_list)
        else:
            postprocess_file_num = len(subgraph_model_buf)

        for i in range(postprocess_file_num):
            if subgraph_model_buf is not None:
                postprocess_converter = tflite_converter.TfliteConverter(postprocess_file_list,bFixed_model,subgraph_model_buf[i],is_decrypt)
            else:
                postprocess_converter = tflite_converter.TfliteConverter(postprocess_file_list[i],bFixed_model,is_decrypt)
            tflite2mace_graph_def_post, list_input_name_post, list_output_name_post = postprocess_converter.run()
            postmodel_list.append(tflite2mace_graph_def_post)
        convertSGS = SGSModel_converter_from_Mace.ConvertSGSModel(tflite2mace_graph_def_model, list_input_name, list_output_name, output_dir, postmodel_list, mode, None, platform='tflite')
    else:
        #convertSGS = SGSModel_converter_tflite.ConvertSGSModel(tflite2mace_graph_def_model, list_input_name, list_output_name, output_dir,convert_kb=convert_kb)
        convertSGS = SGSModel_converter_from_Mace.ConvertSGSModel(tflite2mace_graph_def_model, list_input_name, list_output_name, output_dir, convert_kb=convert_kb, platform='tflite')
    if DUMP_MACE:
          file_name = output_dir.strip("./").split(".")[0]
          f = open(file_name+'_mace.txt', 'w')
          f.write(str(tflite2mace_graph_def_model))
          f.close()

    return convertSGS


