# Tencent is pleased to support the open source community by making TNN available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import caffe2onnx_src.c2oObject as Node
import copy

def getParams(layer):
    param = layer.tile_param
    axis = 1
    if param.HasField('axis'):
        axis = param.axis
    tiles = param.tiles
    return axis, tiles

def getTileOutShape(input_shape, axis, tiles):
    output_shape = copy.deepcopy(input_shape)
    output_shape[axis] = input_shape[axis] * tiles
    return [output_shape]

def createTile(layer, node_name, input_name, output_name, input_shape, axis, tiles):
    output_shape = getTileOutShape(input_shape[0], axis, tiles)
    node = Node.c2oNode(layer, node_name, "Tile", input_name, output_name, input_shape, output_shape, Flag=True)
    return node
