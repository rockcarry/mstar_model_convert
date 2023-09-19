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


import math

import numpy as np
import six
import copy
import pdb
from mace.python.tools.converter_tool.transformer import Transformer
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import FilterFormat
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.convert_util import mace_check


class ShapeInference(object):
    """Currently we only use it to infer caffe shape, we use tensorflow engine
    to infer tensorflow op shapes, since tensorflow has too many ops."""

    def __init__(self, net, input_nodes):
        self._op_shape_inference = {
            MaceOp.Conv2D.name: self.infer_shape_conv_pool_shape,
            MaceOp.Deconv2D.name: self.infer_shape_deconv,
            MaceOp.DepthwiseConv2d.name: self.infer_shape_conv_pool_shape,
            MaceOp.DepthwiseDeconv2d.name: self.infer_shape_deconv,
            MaceOp.Eltwise.name: self.infer_shape_general,
            MaceOp.BatchNorm.name: self.infer_shape_general,
            MaceOp.AddN.name: self.infer_shape_general,
            MaceOp.Activation.name: self.infer_shape_general,
            MaceOp.Pooling.name: self.infer_shape_conv_pool_shape,
            MaceOp.Concat.name: self.infer_shape_concat,
            MaceOp.Slice.name: self.infer_shape_slice,
            MaceOp.Split.name: self.infer_shape_split,
            MaceOp.Softmax.name: self.infer_shape_general,
            MaceOp.FullyConnected.name: self.infer_shape_fully_connected,
            MaceOp.Crop.name: self.infer_shape_crop,
            MaceOp.BiasAdd.name: self.infer_shape_general,
            MaceOp.ChannelShuffle.name: self.infer_shape_channel_shuffle,
            MaceOp.Transpose.name: self.infer_shape_permute,
            MaceOp.PriorBox.name: self.infer_shape_prior_box,
            MaceOp.Reshape.name: self.infer_shape_reshape,
            #add by sigmastar
            'ArgMax':self.infer_shape_ArgMax,
            'Axpy':self.infer_shape_Axpy,
            'CReLU':self.infer_shape_CReLU,
            'Clip':self.infer_shape_general,
            'Normalize': self.infer_shape_Normalize,
            'Reorg': self.infer_shape_Reorg,
            'Upsample': self.infer_shape_Upsample,
            'Dropout': self.infer_shape_Dropout,
            'LSTM': self.infer_shape_LSTM,
            'Threshold': self.infer_shape_Threshold,
            'Tile': self.infer_shape_Tile,
            'Scale':self.infer_shape_Scale,
            'Reverse':self.infer_shape_Reverse,
            'ContinuationIndicator':self.infer_shape_ContinuationIndicator,
            'ROIPooling':self.infer_shape_ROIPooling,
            'Power':self.infer_shape_Power,
            'SGS_SSD_Postprocess': self.infer_shape_SGS_CAFFE_SSD_Postprocess,
            'SGS_YoloV2_Postprocess': self.infer_shape_SGS_YoloV2_Postprocess,
            'SGS_YoloV3_Postprocess': self.infer_shape_SGS_YoloV3_Postprocess,
            'SGS_LanceNet_Postprocess': self.infer_shape_SGS_LanceNet_Postprocess,
            'SGS_FDA_Postprocess': self.infer_shape_SGS_FDA_Postprocess,
            'PriorBox_RFC': self.infer_shape_prior_box_RFC,
        }

        self._net = net
        self._output_shape_cache = {}
        for input_node in input_nodes:
            input_shape = input_node.shape[:]
            # transpose input from NCHW to NHWC
            #Transformer.transpose_shape(input_shape, [0, 3, 1, 2])
            self._output_shape_cache[input_node.name] = input_shape
        for tensor in net.tensors:
            self._output_shape_cache[tensor.name] = list(tensor.dims)

    def run(self):
        for op in self._net.op:
            mace_check(op.type in self._op_shape_inference,
                       "Mace does not support caffe op type %s yet"
                       % op.type)
            print("Begin to inference shape ",op.name)
            self._op_shape_inference[op.type](op)

    def add_output_shape(self, op, shapes):
        mace_check(len(op.output) == len(shapes),
                   "Op %s (%s) output count is different from "
                   "output shape count" % (
                       op.name, op.type))
        for i in six.moves.range(len(shapes)):
            output_name = op.output[i]
            output_shape = op.output_shape.add()
            output_shape.dims.extend(shapes[i])
            self._output_shape_cache[output_name] = shapes[i]

    def infer_shape_general(self, op):
        if len(op.input) > 0:
            mace_check(op.input[0] in self._output_shape_cache,
                       "Op %s input %s does not exist"
                       % (op.name, op.input[0]))
            input_shape = self._output_shape_cache[op.input[0]]
            self.add_output_shape(op, [input_shape])

    def infer_shape_conv_pool_shape(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        output_shape = np.zeros_like(input_shape)
        if op.type == MaceOp.Pooling:
            filter_shape = list(
                ConverterUtil.get_arg(op, MaceKeyword.mace_kernel_str).ints)
            if ConverterUtil.data_format(op) == DataFormat.NCHW:
                filter_shape = [input_shape[1], input_shape[1]] + filter_shape
                if ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_global_pooling_str) \
                        is not None:
                    filter_shape[2] = input_shape[2]
                    filter_shape[3] = input_shape[3]
            else:  # NHWC
                filter_shape = filter_shape + [input_shape[1], input_shape[1]]
                if ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_global_pooling_str) \
                        is not None:
                    filter_shape[0] = input_shape[1]
                    filter_shape[1] = input_shape[2]
        else:
            filter_shape = self._output_shape_cache[op.input[1]]
        paddings = ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_padding_values_str).ints  # noqa
        strides = ConverterUtil.get_arg(op, MaceKeyword.mace_strides_str).ints
        dilations_arg = ConverterUtil.get_arg(op,
                                              MaceKeyword.mace_dilations_str)
        if dilations_arg is not None:
            dilations = dilations_arg.ints
        else:
            dilations = [1, 1]
        if op.type == MaceOp.Pooling and (ConverterUtil.get_arg(op,'ceil_mode').i == 1):
            round_func = math.ceil
        else:
            round_func = math.floor

        output_shape[0] = input_shape[0]
        if ConverterUtil.data_format(op) == DataFormat.NCHW \
                and ConverterUtil.filter_format(self._net) == FilterFormat.OIHW:  # noqa
            # filter format: OIHW
            if op.type == MaceOp.DepthwiseConv2d.name:
                output_shape[1] = filter_shape[0] * filter_shape[1]
            else:
                output_shape[1] = filter_shape[0]
            output_shape[2] = int(
                round_func((input_shape[2] + (paddings[2]+paddings[3]) - filter_shape[2] -
                            (filter_shape[2] - 1) *
                            (dilations[0] - 1)) / float(strides[0]))) + 1
            output_shape[3] = int(
                round_func((input_shape[3] + (paddings[0]+paddings[1]) - filter_shape[3] -
                            (filter_shape[3] - 1) *
                            (dilations[1] - 1)) / float(strides[1]))) + 1
        else:
            mace_check(False,
                       "Mace can only infer shape for"
                       " NCHW input and OIHW filter")
        output_shape_array = []
        for i in six.moves.range(len(op.output)):
            output_shape_array.append(output_shape)
        self.add_output_shape(op, output_shape_array)

    def infer_shape_deconv(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        output_shape = np.zeros_like(input_shape)
        filter_shape = self._output_shape_cache[op.input[1]]
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'num_output':
            num_output = arg[i].i

        paddings = ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_padding_values_str).ints  # noqa
        strides = ConverterUtil.get_arg(op, MaceKeyword.mace_strides_str).ints
        dilations_arg = ConverterUtil.get_arg(op,
                                              MaceKeyword.mace_dilations_str)
        if dilations_arg is not None:
            dilations = dilations_arg.ints
        else:
            dilations = [1, 1]
        round_func = math.floor

        group_arg = ConverterUtil.get_arg(op,
                                          MaceKeyword.mace_group_str)

        output_shape[0] = input_shape[0]
        if ConverterUtil.data_format(op) == DataFormat.NCHW \
                and ConverterUtil.filter_format(self._net) == FilterFormat.OIHW:  # noqa
            # filter format: IOHW
            if op.type == MaceOp.DepthwiseDeconv2d.name:
                    output_shape[1] = filter_shape[0] * filter_shape[1]
            else:
                if group_arg != None and group_arg.i > 1 :
                    output_shape[1] = num_output
                else:
                    output_shape[1] = filter_shape[0]
            output_shape[2] = int(
                round_func((input_shape[2] - 1) * strides[0] +
                           (filter_shape[2] - 1) * (dilations[0] - 1) +
                           filter_shape[2] - (paddings[2]+paddings[3])))
            output_shape[3] = int(
                round_func((input_shape[3] - 1) * strides[1] +
                           (filter_shape[3] - 1) * (dilations[1] - 1) +
                           filter_shape[3] - (paddings[0]+paddings[1])))
        else:
            mace_check(False,
                       "Mace can only infer shape for"
                       " NCHW input and OIHW filter")
        print("deconv layer %s (%s) input:%s filter:%s output:%s" %
              (op.name, op.type, input_shape, filter_shape, output_shape))

        self.add_output_shape(op, [output_shape])

    def infer_shape_concat(self, op):
        output_shape = list(self._output_shape_cache[op.input[0]])
        axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        if axis < 0:
            axis = len(output_shape) + axis
        output_shape[axis] = 0
        for input_node in op.input:
            input_shape = list(self._output_shape_cache[input_node])
            output_shape[axis] = output_shape[axis] + input_shape[axis]
        self.add_output_shape(op, [output_shape])

    def infer_shape_slice(self, op):
        output_shape = copy.deepcopy(self._output_shape_cache[op.input[0]])
        axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        output_shapes = []
        if ConverterUtil.get_arg(op, 'slice_point') != None:
            slice_point = ConverterUtil.get_arg(op, 'slice_point').ints
            slice = []
            prev = 0
            for i in range(len(slice_point)):
              slice.append(slice_point[i] - prev)
              prev = slice_point[i]
            slice.append(output_shape[axis] - prev)
            for index,_ in enumerate(op.output):
                output_shape_one = output_shape
                output_shape_one[axis] = slice[index]
                output_tem = copy.deepcopy(output_shape_one)
                output_shapes.append(output_tem)
        else:
            output_shape[axis] = output_shape[axis] // len(op.output)
            for _ in op.output:
                output_shapes.append(output_shape)
        self.add_output_shape(op, output_shapes)

    def infer_shape_split(self, op):
        output_shape = self._output_shape_cache[op.input[0]]
        output_shapes = []
        for _ in op.output:
            output_shapes.append(output_shape)
        self.add_output_shape(op, output_shapes)

    def infer_shape_fully_connected(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        weight_shape = self._output_shape_cache[op.input[1]]
        axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        if ConverterUtil.data_format(op) == DataFormat.NCHW:
          if len(input_shape) == 4:
            output_shape = [input_shape[0], weight_shape[0], 1, 1]
          else:
            innershape = 1
            for i in six.moves.range(len(input_shape)-axis):
               innershape *= input_shape[axis+i]
            if input_shape[-1] != innershape:
               output_shape = []
               for i in six.moves.range(len(input_shape)-axis):
                  output_shape.append(input_shape[i])
               output_shape[axis] = weight_shape[0]
            else:
               output_shape = copy.deepcopy(input_shape)
               output_shape[-1] = weight_shape[0]
        else:
            mace_check(False, "format %s is not supported"
                       % ConverterUtil.data_format(op))
        self.add_output_shape(op, [output_shape])

    def infer_shape_crop(self, op):
        mace_check(len(op.input) == 2, "crop layer needs two inputs")
        output_shape = self._output_shape_cache[op.input[0]]
        input1_shape = self._output_shape_cache[op.input[1]]
        axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        if axis < 0:
            axis = len(output_shape) + axis
        for i in range(len(output_shape)):
            if i >= axis:
                output_shape[i] = input1_shape[i]
        self.add_output_shape(op, [output_shape])


    def infer_shape_channel_shuffle(self, op):
        output_shape = self._output_shape_cache[op.input[0]]
        self.add_output_shape(op, [output_shape])

    def infer_shape_permute(self, op):
        output_shape = list(self._output_shape_cache[op.input[0]])
        dims = ConverterUtil.get_arg(op, MaceKeyword.mace_dims_str).ints
        for i in six.moves.range(len(dims)):
            output_shape[i] = self._output_shape_cache[op.input[0]][dims[i]]
        self.add_output_shape(op, [output_shape])

    def infer_shape_prior_box(self, op):
        output_shape = [1, 1, 1]
        input_shape = list(self._output_shape_cache[op.input[0]])
        input_w = input_shape[3]
        input_h = input_shape[2]
        min_size = ConverterUtil.get_arg(op, MaceKeyword.mace_min_size_str).floats  # noqa
        max_size = ConverterUtil.get_arg(op, MaceKeyword.mace_max_size_str).floats  # noqa
        aspect_ratio = ConverterUtil.get_arg(op, MaceKeyword.mace_aspect_ratio_str).floats  # noqa
        num_prior = len(aspect_ratio) * len(min_size) + len(max_size)

        output_shape[2] = num_prior * input_h * input_w * 4
        self.add_output_shape(op, [output_shape])

    def infer_shape_prior_box_RFC(self, op):
        output_shape = [1, 1, 1]
        input_shape = list(self._output_shape_cache[op.input[0]])
        input_w = input_shape[3]
        input_h = input_shape[2]
        min_size = ConverterUtil.get_arg(op, MaceKeyword.mace_min_size_str).floats  # noqa
        max_size = ConverterUtil.get_arg(op, MaceKeyword.mace_max_size_str).floats  # noqa
        aspect_ratio = ConverterUtil.get_arg(op, MaceKeyword.mace_aspect_ratio_str).floats  # noqa
        num_prior = len(aspect_ratio) * 2 + 1
        output_shape[2] = num_prior * input_h * input_w * 4
        self.add_output_shape(op, [output_shape])

    def infer_shape_reshape(self, op):
     if ConverterUtil.get_arg(op, MaceKeyword.mace_end_axis_str) is not None: #flatten
        output_shape = []
        axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        end_axis = ConverterUtil.get_arg(op, MaceKeyword.mace_end_axis_str).i  # noqa
        end_axis = end_axis if end_axis > 0 else end_axis + len(
            list(self._output_shape_cache[op.input[0]]))
        dim = 1
        for i in range(0, axis):
            output_shape.append(self._output_shape_cache[op.input[0]][i])
        for i in six.moves.range(axis, end_axis + 1):
            dim *= self._output_shape_cache[op.input[0]][i]
        output_shape.append(-1)
        for i in six.moves.range(end_axis + 1, len(
                list(self._output_shape_cache[op.input[0]]))):
            output_shape.append(self._output_shape_cache[op.input[0]][i])
        output_shape[axis] = dim
        self.add_output_shape(op, [output_shape])
     else: #reshape
        input_shape_dims = len(list(self._output_shape_cache[op.input[0]]))
        input_start_axis = ConverterUtil.get_arg(op, 'reshape_' + MaceKeyword.mace_axis_str).i
        start_axis = input_start_axis if input_start_axis >= 0 \
                    else (input_shape_dims + input_start_axis + 1)
        num_axes = ConverterUtil.get_arg(op, MaceKeyword.mace_num_axes_str).i
        end_axis = input_shape_dims if num_axes == -1 \
                    else (start_axis + num_axes)
        dim = ConverterUtil.get_arg(op, MaceKeyword.mace_dim_str).ints

        product = input_size = 1
        copy_axes = []
        inferred_axis = -1
        top_shape_index = 0
        bottom_shape_index = 0
        copy_axis_index = []
        output_shape = []

        for i in six.moves.range(input_shape_dims):
            input_size *= self._output_shape_cache[op.input[0]][i]

        for i in six.moves.range(start_axis):
            output_shape.append(self._output_shape_cache[op.input[0]][i])
            top_shape_index += 1
            bottom_shape_index += 1
            product *= self._output_shape_cache[op.input[0]][i]

        for i in six.moves.range(len(dim)):
            if dim[i] == 0:
                copy_axes.append(top_shape_index)
                copy_axis_index.append(i)
                output_shape.append(1)
                #product *= self._output_shape_cache[op.input[0]][top_shape_index]
                top_shape_index += 1
            elif dim[i] == -1:
                inferred_axis = top_shape_index
                output_shape.append(1)
                top_shape_index += 1
            else:
                output_shape.append(dim[i])
                product *= dim[i]
                top_shape_index += 1

        bottom_shape_index += num_axes
        for i in six.moves.range(end_axis, input_shape_dims):
            output_shape.append(self._output_shape_cache[op.input[0]][bottom_shape_index])
            product *= self._output_shape_cache[op.input[0]][bottom_shape_index]
            top_shape_index += 1
            bottom_shape_index += 1

        if copy_axes != []:
          for i in six.moves.range(len(copy_axes)):
            mace_check (input_shape_dims > (start_axis + copy_axis_index[i]), \
                "new shape contains a 0, but there was no corresponding bottom axis to copy Please check {} args"\
                .format(op.name))
            output_shape[copy_axes[i]] = self._output_shape_cache[op.input[0]][copy_axes[i]]
            product *= self._output_shape_cache[op.input[0]][copy_axes[i]]
        if inferred_axis != -1:
            output_shape[inferred_axis] = int(input_size / product)
        self.add_output_shape(op, [output_shape])
        '''
        if ConverterUtil.get_arg(op, MaceKeyword.mace_dim_str) is not None:
            dim = ConverterUtil.get_arg(op, MaceKeyword.mace_dim_str).ints
            output_shape = list(dim)
            product = input_size = 1
            idx = -1
            for i in six.moves.range(len(self._output_shape_cache[op.input[0]])):
                input_size *= self._output_shape_cache[op.input[0]][i]
            for i in six.moves.range(len(dim)):
                if dim[i] == 0:
                    output_shape[i] = self._output_shape_cache[op.input[0]][i]
                    product *= self._output_shape_cache[op.input[0]][i]
                elif dim[i] == -1:
                    idx = i
                    output_shape[i] = 1
                else:
                    output_shape[i] = dim[i]
                    product *= dim[i]
            if idx != -1:
                output_shape[idx] = int(input_size / product)
            self.add_output_shape(op, [output_shape])
        else:
            output_shape = []
            axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
            end_axis = ConverterUtil.get_arg(op, MaceKeyword.mace_end_axis_str).i  # noqa
            end_axis = end_axis if end_axis > 0 else end_axis + len(
                list(self._output_shape_cache[op.input[0]]))
            dim = 1
            for i in range(0, axis):
                output_shape.append(self._output_shape_cache[op.input[0]][i])
            for i in six.moves.range(axis, end_axis + 1):
                dim *= self._output_shape_cache[op.input[0]][i]
            output_shape.append(-1)
            for i in six.moves.range(end_axis + 1, len(
                    list(self._output_shape_cache[op.input[0]]))):
                output_shape.append(self._output_shape_cache[op.input[0]][i])
            output_shape[axis] = dim
            self.add_output_shape(op, [output_shape])
        '''
    def infer_shape_ArgMax(self, op):
        output_shape = list(self._output_shape_cache[op.input[0]])
        axis = ConverterUtil.get_arg(op, 'axis').i
        if axis < 0:
            axis = len(output_shape) + axis
        input_shape = output_shape
        output_shape[axis] = 1
        self.add_output_shape(op, [output_shape])

    def infer_shape_Axpy(self, op):
         '''
         * @param Formulation:
         *            F = a * X + Y
         *    Shape info:
         *            a:  N x C          --> bottom[0]
         *            X:  N x C x H x W  --> bottom[1]
         *            Y:  N x C x H x W  --> bottom[2]
         *            F:  N x C x H x W  --> top[0]
         '''
         if len(op.input) > 0:
            mace_check(op.input[2] in self._output_shape_cache,
                       "Op %s input %s does not exist"
                       % (op.name, op.input[2]))
            input_shape = self._output_shape_cache[op.input[2]]
            self.add_output_shape(op, [input_shape])

    def infer_shape_CReLU(self, op):
        output_shape = list(self._output_shape_cache[op.input[0]])
        axis = ConverterUtil.get_arg(op, 'concat_axis').i
        if axis < 0:
            axis = len(output_shape) + axis
        input_shape = output_shape
        output_shape[axis] = input_shape[axis] + input_shape[axis]
        self.add_output_shape(op, [output_shape])
    def infer_shape_Normalize(self, op):
        output_shape = self._output_shape_cache[op.input[0]]
        self.add_output_shape(op, [output_shape])
    def infer_shape_Reorg(self, op):
        #only support stride is 2
        output_shape = self._output_shape_cache[op.input[0]]
        input_shape = list(self._output_shape_cache[op.input[0]])
        input_n = input_shape[0]
        input_c = input_shape[1]
        input_h = input_shape[2]
        input_w = input_shape[3]
        output_shape = [input_n,int(input_c*4),int(input_h/2),int(input_w/2)]
        self.add_output_shape(op, [output_shape])
    def infer_shape_Upsample(self, op):
        output_shape = self._output_shape_cache[op.input[0]]
        input_shape = list(self._output_shape_cache[op.input[0]])
        scale = ConverterUtil.get_arg(op,'scale').i
        input_n = input_shape[0]
        input_c = input_shape[1]
        input_h = input_shape[2]
        input_w = input_shape[3]
        output_shape = [input_n,int(input_c),int(input_h*scale),int(input_w*scale)]
        self.add_output_shape(op, [output_shape])
    def infer_shape_Dropout(self, op):
        if len(op.input) > 0:
            mace_check(op.input[0] in self._output_shape_cache,
                       "Op %s input %s does not exist"
                       % (op.name, op.input[0]))
            input_shape = self._output_shape_cache[op.input[0]]
            self.add_output_shape(op, [input_shape])
    def infer_shape_LSTM(self, op):
        indicator_shape = self._output_shape_cache[op.input[1]]
        num_output = ConverterUtil.get_arg(op,'num_output').i
        output_shape = [indicator_shape[0],1,num_output]
        self.add_output_shape(op, [output_shape])

    def infer_shape_Threshold(self, op):
        if len(op.input) > 0:
            mace_check(op.input[0] in self._output_shape_cache,
                       "Op %s input %s does not exist"
                       % (op.name, op.input[0]))
            input_shape = self._output_shape_cache[op.input[0]]
            self.add_output_shape(op, [input_shape])

    def infer_shape_Scale(self, op):
        if len(op.input) > 0:
            mace_check(op.input[0] in self._output_shape_cache,
                       "Op %s input %s does not exist"
                       % (op.name, op.input[0]))
            input_shape = self._output_shape_cache[op.input[0]]
            self.add_output_shape(op, [input_shape])

    def infer_shape_Reverse(self, op):
        if len(op.input) > 0:
            mace_check(op.input[0] in self._output_shape_cache,
                       "Op %s input %s does not exist"
                       % (op.name, op.input[0]))
            input_shape = self._output_shape_cache[op.input[0]]
            self.add_output_shape(op, [input_shape])

    def infer_shape_ContinuationIndicator(self, op):
        time_step = ConverterUtil.get_arg(op,'time_step').i
        mini_batch = ConverterUtil.get_arg(op,'batch_size').i
        output_shape = [time_step,mini_batch]
        #output_shape = [time_step]
        self.add_output_shape(op, [output_shape])

    def infer_shape_Tile(self, op):
        output_shape = self._output_shape_cache[op.input[0]]
        input_shape = list(self._output_shape_cache[op.input[0]])
        tiles = ConverterUtil.get_arg(op,'tiles').i
        axis = ConverterUtil.get_arg(op,'axis').i
        output_shape[axis] = input_shape[axis] * tiles
        self.add_output_shape(op, [output_shape])

    def infer_shape_ROIPooling(self, op):
        pooled_w = ConverterUtil.get_arg(op,'pooled_w').i
        pooled_h = ConverterUtil.get_arg(op,'pooled_h').i
        input_shape_0 = list(self._output_shape_cache[op.input[0]])
        input_shape_1 = list(self._output_shape_cache[op.input[1]])
        if len(input_shape_0) == 4:
          channel = input_shape_0[1]
          num = input_shape_1[0]
        else:
          channel = input_shape_1[1]
          num = input_shape_0[0]
        output_shape = [num,channel,pooled_h,pooled_w]
        self.add_output_shape(op, [output_shape])

    def infer_shape_Power(self, op):
        if len(op.input) > 0:
            mace_check(op.input[0] in self._output_shape_cache,
                       "Op %s input %s does not exist"
                       % (op.name, op.input[0]))
            input_shape = self._output_shape_cache[op.input[0]]
            self.add_output_shape(op, [input_shape])

    def infer_shape_SGS_SSD_Postprocess(self, op):
        self.add_output_shape(op, [[1,10,4],[1,10],[1,10],[1]])
        pass
    def infer_shape_SGS_YoloV2_Postprocess(self, op):
        self.add_output_shape(op, [[1,100,4],[1,100],[1,100],[1]])
        pass
    def infer_shape_SGS_YoloV3_Postprocess(self, op):
        self.add_output_shape(op, [[1,100,4],[1,100],[1,100],[1]])
        pass
    def infer_shape_SGS_LanceNet_Postprocess(self, op):
        self.add_output_shape(op, [[1,100,4],[1,100],[1,100],[1]])
        pass
    def infer_shape_SGS_FDA_Postprocess(self, op):
        self.add_output_shape(op, [[1,100,4],[1,100,10],[1,100],[1]])
        pass
    def infer_shape_SGS_CAFFE_SSD_Postprocess(self, op):
        if op.name == 'TFLite_RFCSSD_Detection_PostProcess':
            self.add_output_shape(op, [[1,100,4],[1,100],[1,100],[1]])
        else:
            self.add_output_shape(op, [[1,10,4],[1,10],[1,10],[1]])
        pass
