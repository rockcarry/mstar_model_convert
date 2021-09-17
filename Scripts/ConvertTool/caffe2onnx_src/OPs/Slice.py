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

def analyzeLayer(layer, input_shape):
    # 获取到 slice_point
    axis = layer.slice_param.axis
    starts = [0]
    axes = [axis]
    for step in layer.slice_param.slice_point:
        starts.append(step)
        axes.append(axis)
    # 获取需要进行操作的轴
    ends = []
    for step in layer.slice_param.slice_point:
        ends.append(step)
    # 这个地方搞了一个小 trick, 使用输入的 axis 作为最后一个
    ends.append(input_shape[0][axis])
    return starts, ends, axes


def getSliceOutShape(input_shape, start, end):
    if len(input_shape[0]) == 4:
        output_shape = [[input_shape[0][0], end - start, input_shape[0][2], input_shape[0][3]]]
    elif len(input_shape[0]) == 2:
        output_shape = [[input_shape[0][0], end - start]]
    else:
        print("Unsupport slice shape")
        exit(-1)

    return output_shape



# 构建节点
def createSlice(layer, node_name, input_name, output_name, input_shape, start, end):

    output_shape = getSliceOutShape(input_shape, start, end)

    node = Node.c2oNode(layer, node_name, "Slice", input_name, output_name, input_shape, output_shape, Flag=True)
    return node
