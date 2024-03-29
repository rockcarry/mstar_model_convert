﻿# Copyright 2018 The MACE Authors. All Rights Reserved.
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


import sys
from enum import Enum
import six
import pdb
import os
from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool.base_converter import PoolingType
from mace.python.tools.converter_tool.base_converter import PaddingMode
from mace.python.tools.converter_tool.base_converter import ActivationType
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import ReduceType
from mace.python.tools.converter_tool.base_converter import FrameworkType
from mace.python.tools.converter_tool.base_converter import RoundMode
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import FilterFormat
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.converter_tool.base_converter import PadType

from mace.python.tools.convert_util import *
from mace.python.tools.onnx import symbolic_shape_infer
from mace.python.tools.onnx import onnx_simplifier

import numpy as np
import math
import onnx
import onnx.utils
from onnx import mapping, numpy_helper, TensorProto
from numbers import Number


IS_PYTHON3 = sys.version_info > (3,)


class AttributeType(Enum):
    INT = 100
    FLOAT = 101
    INTS = 102
    FLOATS = 103
    BOOL = 104


OnnxSupportedOps = [
    'Abs',
    # 'Acos',
    # 'Acosh',
    'Add',
    'Affine',
    # 'And',
    'Append',
    'ArgMax',
    'ArgMin',
    # 'Asin',
    # 'Asinh',
    # 'Atan',
    # 'Atanh',
    'AveragePool',
    'BatchNormalization',
    'BatchNorm',
    'Cast',
    # 'Ceil',
    'Clip',
    # 'Compress',
    'Concat',
    'Constant',
    # 'ConstantLike',
    'Conv',
    'ConvTranspose',
    # 'Cos',
    # 'Cosh',
    'DepthToSpace',
    'DimRange',
    'Div',
    'Dropout',
    'DynamicLSTM',
    # 'Elu',
    'Equal',
    'Exp',
    'Expand',
    'ExtractPooling',
    # 'EyeLike',
    'Flatten',
    # 'Floor',
    # 'GRU',
    'Gather',
    'Gemm',
    'GlobalAveragePool',
    # 'GlobalLpPool',
    'GlobalMaxPool',
    # 'Greater',
    # 'HardSigmoid',
    # 'Hardmax',
    'Identity',
    # 'If',
    'IfDefined',
    'ImageScaler',
    # 'InstanceNormalization',
    # 'LRN',
    'LSTM',
    'LstmNonlinear',
    'LeakyRelu',
    # 'Less',
    # 'Log',
    'LogSoftmax',
    # 'Loop',
    'LpNormalization',
    # 'LpPool',
    'MatMul',
    'Max',
    'MaxPool',
    # 'MaxRoiPool',
    # 'MaxUnpool',
    # 'Mean',
    'Min',
    'Mul',
    # 'Multinomial',
    'Neg',
    'Normalize',
    # 'Not',
    'Offset',
    # 'OneHot',
    # 'Or',
    'PRelu',
    'Pad',
    'PadContext',
    'PNorm',
    'Pow',
    # 'RNN',
    # 'RandomNormal',
    # 'RandonNormalLike',
    # 'RandonUniform',
    # 'RandonUniformLike',
    'Reciprocal',
    # 'ReduceL1',
    # 'ReduceL2',
    # 'ReduceLogSum',
    # 'ReduceLogSumExp',
    'ReduceMax',
    'ReduceMean',
    'ReduceMin',
    'ReduceProd',
    'ReduceSum',
    # 'ReduceSumSquare',
    'Relu',
    'ReplaceIndex',
    'Reshape',
    'Round',
    'Resize',
    'Scale',
    # 'Scan',
    # 'Selu',
    'Shape',
    'Sigmoid',
    # 'Sin',
    # 'Sinh',
    # 'Size',
    'Slice',
    'Softmax',
    # 'Softplus',
    # 'Softsign',
    'SpaceToDepth',
    'Splice',
    'Split',
    'Sqrt',
    'Squeeze',
    'Sub',
    'Subsample',
    'Sum',
    'SumGroup',
    # 'Tan',
    'Tanh',
    'TargetRMSNorm',
    'Tile',
    # 'TopK',
    'Transpose',
    'Unsqueeze',
    'Upsample',
    # 'Xor',
]

OnnxOpType = Enum('OnnxOpType',
                  [(op, op) for op in OnnxSupportedOps],
                  type=str)

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}


def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)


def convert_onnx(attr):
    return convert_onnx_attribute_proto(attr)


def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return str(attr_proto.s, 'utf-8')\
            if IS_PYTHON3 else attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        if IS_PYTHON3:
            str_list = map(lambda x: str(x, 'utf-8'), str_list)
        return str_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]


class OnnxNode(object):
    def __init__(self, node):
        self.name = str(node.name)
        if self.name == '':
            self.name = str(node.output)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name,
                            translate_onnx(attr.name, convert_onnx(attr)))
                           for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node

    def print_info(self):
        print("node: ", self.name)
        print("    type: ", self.op_type)
        print("    domain: ", self.domain)
        print("    inputs: ", self.inputs)
        print("    outputs: ", self.outputs)
        print("    attrs:")
        for arg in self.attrs:
            print("        %s: %s" % (arg, self.attrs[arg]))


class OnnxTensor(object):
    def __init__(self, name, value, shape, dtype):
        self._name = name
        self._tensor_data = value
        self._shape = shape
        self._dtype = dtype


class OnnxConverter(base_converter.ConverterInterface):
    pooling_type_mode = {
        OnnxOpType.AveragePool.name: PoolingType.AVG,
        OnnxOpType.MaxPool.name: PoolingType.MAX,
        OnnxOpType.GlobalAveragePool.name: PoolingType.GlobalAveragePool,
        OnnxOpType.GlobalMaxPool.name: PoolingType.GlobalMaxPool
    }

    auto_pad_mode = {
        'NOTSET': PaddingMode.NA,
        'SAME_UPPER': PaddingMode.SAME,
        'SAME_LOWER': PaddingMode.SAME,
        'VALID': PaddingMode.VALID,
    }
    auto_pad_mode = {six.b(k): v for k, v in six.iteritems(auto_pad_mode)}

    eltwise_type = {
        OnnxOpType.Mul.name: EltwiseType.PROD,
        OnnxOpType.Add.name: EltwiseType.SUM,
        OnnxOpType.Max.name: EltwiseType.MAX,
        OnnxOpType.Min.name: EltwiseType.MIN,
        OnnxOpType.Abs.name: EltwiseType.ABS,
        OnnxOpType.Pow.name: EltwiseType.POW,
        OnnxOpType.Sub.name: EltwiseType.SUB,
        OnnxOpType.Div.name: EltwiseType.DIV,
        OnnxOpType.Neg.name: EltwiseType.NEG,
        OnnxOpType.Sum.name: EltwiseType.SUM,
        OnnxOpType.Equal.name: EltwiseType.EQUAL,
        OnnxOpType.Sqrt.name: EltwiseType.POW,
        OnnxOpType.Reciprocal.name: EltwiseType.POW,
        OnnxOpType.Scale.name: EltwiseType.PROD,
        OnnxOpType.Clip.name: EltwiseType.CLIP,
    }

    reduce_type = {
        OnnxOpType.GlobalAveragePool.name: ReduceType.MEAN,
        OnnxOpType.GlobalMaxPool.name: ReduceType.MAX,
        OnnxOpType.ReduceMax.name: ReduceType.MAX,
        OnnxOpType.ReduceMean.name: ReduceType.MEAN,
        OnnxOpType.ReduceMin.name: ReduceType.MIN,
        OnnxOpType.ReduceProd.name: ReduceType.PROD,
        OnnxOpType.ReduceSum.name: ReduceType.SUM,
    }

    activation_type = {
        OnnxOpType.Relu.name: ActivationType.RELU,
        OnnxOpType.LeakyRelu.name: ActivationType.LEAKYRELU,
        OnnxOpType.PRelu.name: ActivationType.PRELU,
        OnnxOpType.Tanh.name: ActivationType.TANH,
        OnnxOpType.Sigmoid.name: ActivationType.SIGMOID,
    }

    def __init__(self, option,input_shapes,src_model_file,rawdata=False):
        self._op_converters = {
            OnnxOpType.Abs.name: self.convert_eltwise,
            OnnxOpType.Add.name: self.convert_eltwise,
            OnnxOpType.Affine.name: self.convert_affine,
            OnnxOpType.Append.name: self.convert_concat,
            OnnxOpType.ArgMax.name: self.convert_argmax,
            OnnxOpType.ArgMin.name: self.convert_argmax,
            OnnxOpType.AveragePool.name: self.convert_pooling,
            OnnxOpType.BatchNormalization.name: self.convert_fused_batchnorm,
            OnnxOpType.BatchNorm.name: self.convert_fused_batchnorm,
            OnnxOpType.Cast.name: self.convert_cast,
            OnnxOpType.Clip.name: self.convert_clip,
            OnnxOpType.Concat.name: self.convert_concat,
            OnnxOpType.Conv.name: self.convert_conv2d,
            OnnxOpType.ConvTranspose.name: self.convert_deconv,
            OnnxOpType.Constant.name: self.convert_constant,
            OnnxOpType.DepthToSpace.name: self.convert_depth_space,
            OnnxOpType.Dropout.name: self.convert_dropout,
            OnnxOpType.DimRange.name: self.convert_dim_range,
            OnnxOpType.Div.name: self.convert_eltwise,
            OnnxOpType.Equal.name: self.convert_eltwise,
            OnnxOpType.ExtractPooling.name: self.convert_extract_pooling,
            OnnxOpType.Flatten.name: self.convert_flatten,
            OnnxOpType.Gather.name: self.convert_gather,
            OnnxOpType.Gemm.name: self.convert_gemm,
            OnnxOpType.GlobalAveragePool.name: self.convert_pooling,
            OnnxOpType.GlobalMaxPool.name: self.convert_pooling,
            OnnxOpType.Identity.name: self.convert_identity,
            OnnxOpType.IfDefined.name: self.convert_ifdefined,
            OnnxOpType.ImageScaler.name: self.convert_imagescaler,
            OnnxOpType.LeakyRelu.name: self.convert_activation,
            OnnxOpType.LogSoftmax.name: self.convert_softmax,
            OnnxOpType.LpNormalization: self.convert_lpnormalization,
            OnnxOpType.LSTM.name: self.convert_lstm,
            OnnxOpType.LstmNonlinear.name: self.convert_lstm_nonlinear,
            OnnxOpType.DynamicLSTM.name: self.convert_dynamic_lstm,
            OnnxOpType.Max.name: self.convert_eltwise,
            OnnxOpType.MaxPool.name: self.convert_pooling,
            OnnxOpType.MatMul.name: self.convert_matmul,
            OnnxOpType.Min.name: self.convert_eltwise,
            OnnxOpType.Mul.name: self.convert_eltwise,
            OnnxOpType.Neg.name: self.convert_eltwise,
            OnnxOpType.Normalize: self.convert_normalize,
            OnnxOpType.Offset.name: self.convert_subsample,
            OnnxOpType.Pad.name: self.convert_pad,
            OnnxOpType.PadContext.name: self.convert_pad_context,
            OnnxOpType.PNorm.name: self.convert_pnorm,
            OnnxOpType.Pow.name: self.convert_eltwise,
            OnnxOpType.PRelu.name: self.convert_activation,
            OnnxOpType.Relu.name: self.convert_activation,
            OnnxOpType.Reshape.name: self.convert_reshape,
            OnnxOpType.Reciprocal.name: self.convert_eltwise,
            OnnxOpType.ReduceMax.name: self.convert_reduce,
            OnnxOpType.ReduceMean.name: self.convert_reduce,
            OnnxOpType.ReduceMin.name: self.convert_reduce,
            OnnxOpType.ReduceProd.name: self.convert_reduce,
            OnnxOpType.ReduceSum.name: self.convert_reduce,
            OnnxOpType.ReplaceIndex.name: self.convert_replaceindex,
            OnnxOpType.Round.name: self.convert_replaceindex,
            OnnxOpType.Scale.name: self.convert_eltwise,
            OnnxOpType.Shape.name: self.convert_shape,
            OnnxOpType.Sigmoid.name: self.convert_activation,
            OnnxOpType.Slice.name: self.convert_slice,
            OnnxOpType.Softmax.name: self.convert_softmax,
            OnnxOpType.SpaceToDepth.name: self.convert_depth_space,
            OnnxOpType.Splice.name: self.convert_splice,
            OnnxOpType.Split.name: self.convert_split,
            OnnxOpType.Sqrt.name: self.convert_eltwise,
            OnnxOpType.Squeeze.name: self.convert_squeeze,
            OnnxOpType.Sub.name: self.convert_eltwise,
            OnnxOpType.Subsample.name: self.convert_subsample,
            OnnxOpType.Sum.name: self.convert_eltwise,
            OnnxOpType.SumGroup.name: self.convert_sum_group,
            OnnxOpType.Tanh.name: self.convert_activation,
            OnnxOpType.TargetRMSNorm: self.convert_target_rms_norm,
            OnnxOpType.Transpose.name: self.convert_transpose,
            OnnxOpType.Tile.name: self.convert_tile,
            OnnxOpType.Unsqueeze.name: self.convert_unsqueeze,
            OnnxOpType.Upsample.name: self.convert_upsample,
            OnnxOpType.Resize.name: self.convert_resize,
            OnnxOpType.Expand.name: self.convert_expand,
            OnnxOpType.Exp.name: self.convert_exp
        }
        self._isKaldi = False
        self._option = option
        self._mace_net_def = mace_pb2.NetDef()
        self._data_format = DataFormat.NCHW
        self.input_node_array = []
        self._rawdata = rawdata
        ConverterUtil.set_filter_format(self._mace_net_def, DataFormat.OIHW)
        ConverterUtil.add_data_format_arg(self._mace_net_def,
                                          self._data_format)
        output_model_name = os.path.join(
            os.path.dirname(src_model_file),
            os.path.basename(src_model_file).replace('.onnx','_refine.onnx'))

        Debug = False#True
        if Debug:
            onnx_model = onnx.load(output_model_name)
        else:
            shapes = input_shapes.split(':')
            if len(shapes) > 1:
                if len(self._option.input_nodes.keys()) != len(shapes):
                    raise ValueError('Convert multi-inputs onnx need num of --input_arrays and --input_shapes are same!')
            input_shapes_list = []
            for num, name in enumerate(self._option.input_nodes.keys()):
                input_shape = '{}:{}'.format(name, shapes[num])
                input_shapes_list.append(input_shape)
            simplp_model = onnx_simplifier.simplifer(src_model_file,output_model_name,input_shapes_list)
            onnx_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(simplp_model,output_model_name)
            opset_import = onnx_model.opset_import
            print("#########################")
            print("onnx model opset version:\n",opset_import[0])
            print("#########################")
        '''
        onnx_model = onnx.load(src_model_file)
        ir_version = onnx_model.ir_version
        opset_imp = onnx_model.opset_import

        onnx.checker.check_model(onnx_model)

        self._isKaldi = False

        polish_available = True
        print("onnx model IR version: ", ir_version)
        for imp in opset_imp:
            domain = imp.domain
            version = imp.version
            print("constains ops domain: ", domain, "version:", version)
            if 'kaldi' in domain:
                polish_available = False
                self._data_format = DataFormat.NONE
                self._isKaldi = True
        if polish_available:
            onnx_model = onnx.utils.polish_model(onnx_model)
        '''
        self._onnx_model = onnx_model
        self._graph_shapes_dict = {}
        self._consts = {}
        self._replace_tensors = {}
        self.input_node_shapes_array = []

    @staticmethod
    def print_graph_info(graph):
        for value_info in graph.value_info:
            print("value info:", value_info)
        for value_info in graph.input:
            print("inputs info:", value_info)
        for value_info in graph.output:
            print("outputs info:", value_info)

    def extract_shape_info(self, graph):
        def extract_value_info(shape_dict, value_info):
            t = tuple([int(dim.dim_value)
                       for dim in value_info.type.tensor_type.shape.dim])
            if t:
               if len(t) != 1 and t[0] == 0:#if n is zero,replace to 1
                  t_list = list(t)
                  t_list[0] = 1
                  t = tuple(t_list)
               shape_dict[value_info.name] = t

        def extract_value_info_2(shape_dict, value_info):
            t = tuple([int(dim)
                       for dim in value_info.dims])
            if t:
                if len(t) != 1 and t[0] == 0:#if n is zero,replace to 1
                  t_list = list(t)
                  t_list[0] = 1
                  t = tuple(t_list)
                shape_dict[value_info.name] = t
                return
            onnx_tensor = numpy_helper.to_array(value_info)
            size = onnx_tensor.size
            shape = onnx_tensor.shape
            if size > 0 and shape == ():#for clipe op case :initializer input para like  array(0., dtype=float32)
                t = tuple([size])
                shape_dict[value_info.name] = t
        for vi in graph.value_info:
            extract_value_info(self._graph_shapes_dict, vi)
        for vi in graph.input:
            extract_value_info(self._graph_shapes_dict, vi)
        for vi in graph.output:
            extract_value_info(self._graph_shapes_dict, vi)
        for vi in graph.initializer:
            extract_value_info_2(self._graph_shapes_dict, vi)

    def add_tensor(self, name, shape, data_type, value):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type

        if tensor.data_type == mace_pb2.DT_INT32:
            tensor.int32_data.extend(value)
        elif tensor.data_type == mace_pb2.DT_FLOAT:
            tensor.float_data.extend(value)
        else:
            mace_check(False, "Not supported tensor type: %s" % name)

    def run(self):
        graph_def = self._onnx_model.graph
        self.extract_shape_info(graph_def)
        self.convert_tensors(graph_def)
        for input_node in self._option.input_nodes.values():
          self.input_node_shapes_array.append(list(self._graph_shapes_dict[input_node.name]))
          self.input_node_array.append(input_node.name)
        self.convert_ops(graph_def)
        return self._mace_net_def,self.input_node_shapes_array

    def add_stride_pad_kernel_arg(self, attrs, op_def):
        if 'strides' in attrs :
            strides = attrs['strides']
            if len(strides) == 2:
                stride = [strides[0], strides[1]]
            else:
                stride = [strides[0], strides[0]]
        else:
            stride = [1, 1]

        strides_arg = op_def.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(stride)

        if 'kernel_shape' in attrs:
            kernel_shape = attrs['kernel_shape']
            if len(kernel_shape) == 2:
                kernel = [kernel_shape[0], kernel_shape[1]]
            else:
                kernel = [1, kernel_shape[0]]
        else:
            kernel = [1,1]
        kernels_arg = op_def.arg.add()
        kernels_arg.name = MaceKeyword.mace_kernel_str
        kernels_arg.ints.extend(kernel)

        # TODO: Does not support AutoPad yet.
        if 'pads' in attrs:
            pads = attrs['pads']
            if len(pads) == 4:
                pad = [pads[1], pads[3], pads[0], pads[2]]
            elif len(pads) == 2:#eca_net
                pad = [pads[0], pads[1], 0, 0]#left, right, top, bottom
            else:
                pad = [0, 0, 0, 0]
            padding_arg = op_def.arg.add()
            padding_arg.name = MaceKeyword.mace_padding_values_str
            padding_arg.ints.extend(pad)
        elif 'auto_pad' in attrs:
            if attrs['auto_pad'] == 'SAME_UPPER':
               if 'dilations' in attrs:
                 dilations = attrs["dilations"]
               else:
                 dilations = [1, 1]
               output_shape = input_shape = self._graph_shapes_dict[op_def.input[0]]
               if op_def.type == 'Pooling':
                 pad = [0, 0, 0, 0]
               else:
                 filter_shape = self._graph_shapes_dict[op_def.input[1]]
                 round_func = math.floor
                 pad= [1,1,1,1]
                 pad[2] = pad[3] = int(round_func((output_shape[2] - 1) *
                                     float(strides[0]) + (filter_shape[2] - 1) * (dilations[0] - 1) +
                                         filter_shape[2] - input_shape[2]) / 2)
                 pad[0] = pad[1] = int(round_func((output_shape[3] - 1) *
                                     float(strides[1]) + (filter_shape[3] - 1) * (dilations[1] - 1) +
                                         filter_shape[3] - input_shape[3]) / 2)
               padding_arg = op_def.arg.add()
               padding_arg.name = MaceKeyword.mace_padding_values_str
               padding_arg.ints.extend(pad)
            elif attrs['auto_pad'] == 'VALID':
               pad = [0, 0, 0, 0]
               padding_arg = op_def.arg.add()
               padding_arg.name = MaceKeyword.mace_padding_values_str
               padding_arg.ints.extend(pad)
            else:
               mace_check(False,
                 "Does not support AutoPad type yet.")
        else:
            pad = [0, 0, 0, 0]
            padding_arg = op_def.arg.add()
            padding_arg.name = MaceKeyword.mace_padding_values_str
            padding_arg.ints.extend(pad)

    def remove_node(self, node):
        input_name = node.inputs[0]
        output_name = node.outputs[0]
        self._replace_tensors[output_name] = input_name

    @staticmethod
    def squeeze_shape(shape, axis):
        new_shape = []
        if len(axis) > 0:
            for i in range(len(shape)):
                if i not in axis:
                    new_shape.append(shape[i])
        else:
            new_shape = shape
        return new_shape

    @staticmethod
    def unsqueeze_shape(shape, axis):
        new_shape = [n for n in shape]
        for n in axis:
            new_shape.insert(n, 1)
        return new_shape

    @staticmethod
    def transpose_const(tensor):
        shape = tensor.dims
        mace_check(len(shape) == 2, "gemm only supports 2-dim input.")
        tensor_data = np.array(tensor.float_data).reshape(
            shape[0], shape[1])
        tensor_data = tensor_data.transpose(1, 0)
        tensor.float_data[:] = tensor_data.flat
        tensor.dims[:] = tensor_data.shape

    def convert_ops(self, graph_def):
        for n in graph_def.node:
            node = OnnxNode(n)
            mace_check(node.op_type in self._op_converters,
                        "Mace does not support onnx op type %s yet"
                        % node.op_type)
            self._op_converters[node.op_type](node)

    def convert_tensors(self, graph_def):
        initializer = graph_def.initializer
        if initializer:
            for init in initializer:
                tensor = self._mace_net_def.tensors.add()
                tensor.name = init.name
                onnx_tensor = numpy_helper.to_array(init)
                tensor.dims.extend(list(init.dims))
                data_type = onnx_dtype(init.data_type)

                if data_type == np.float32 or data_type == np.float64:
                    tensor.data_type = mace_pb2.DT_FLOAT
                    tensor.float_data.extend(
                        onnx_tensor.astype(np.float32).flat)
                elif data_type == np.int64 or data_type == np.int32:
                    tensor.data_type = mace_pb2.DT_INT32
                    tensor.int32_data.extend(
                        onnx_tensor.astype(np.int32).flat)
                else:
                    mace_check(False,
                               "Not supported tensor type: %s" % data_type)
                self._consts[tensor.name] = tensor

    def convert_general_op(self, node, with_shape=True):
        op = self._mace_net_def.op.add()
        op.name = node.name
        for input in node.inputs:
            if input in self._replace_tensors:
                input = self._replace_tensors[input]
            if node.op_type != "LSTM" and self._graph_shapes_dict[input] == (0,):
                continue
            op.input.append(input)
        for output in node.outputs:
            op.output.append(output)
            if with_shape:
                if output in self._graph_shapes_dict:
                    output_shape = op.output_shape.add()
                    shape_info = self._graph_shapes_dict[output]
                    output_shape.dims.extend(shape_info)
        '''
        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.ONNX.value
        '''
        ConverterUtil.add_data_format_arg(op, self._data_format)
        return op

    def convert_activation(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Activation.name
        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b(self.activation_type[node.op_type].name)

        if "alpha" in node.attrs:
            alpha_value = node.attrs["alpha"]
        else:
            if node.op_type == OnnxOpType.LeakyRelu.name:
                alpha_value = 0.01
            else:
                alpha_value = 0
        alpha_arg = op.arg.add()
        alpha_arg.name = MaceKeyword.mace_activation_leakyrelu_coefficient_str
        alpha_arg.f = alpha_value

    def convert_affine(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.MatMul.name
        transpose_b_arg = op.arg.add()
        transpose_b_arg.name = MaceKeyword.mace_transpose_b_str
        transpose_b_arg.i = 1

    def convert_argmax(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.ArgMax.name

        if 'axis' in node.attrs:
            axis_value = node.attrs['axis']
        else:
            axis_value = 0
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value

        if 'keepdims' in node.attrs:
            keepdims = node.attrs['keepdims']
        else:
            keepdims = 1
        keep_dims_arg = op.arg.add()
        keep_dims_arg.name = MaceKeyword.mace_keepdims_str
        keep_dims_arg.i = keepdims

        if node.op_type == OnnxOpType.ArgMin.name:
            min_arg = op.arg.add()
            min_arg.name = MaceKeyword.mace_argmin_str
            min_arg.i = 1

    def convert_biasadd(self, node):
        self.convert_general_op(node)
        op.type = MaceOp.BiasAdd.name

    def convert_cast(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Cast.name

        if 'to' in node.attrs:
            dtype = node.attrs['to']
            if dtype == np.float32 or dtype == np.float64:
                op.output_type.extend([self._option.data_type])
            elif dtype == np.int64 or dtype == np.int32:
                op.output_type.extend([mace_pb2.DT_INT32])
            else:
                mace_check(False, "data type %s not supported" % dtype)
        else:
            op.output_type.extend([self._option.data_type])

    def convert_concat(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Concat.name
        if self._isKaldi is False:
            mace_check('axis' in node.attrs,
                       'Concat op should have axis attribute.')
            axis_value = node.attrs['axis']
        else:
            axis_value = -1
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value

    def convert_constant(self, node):
        output_name = node.outputs[0]
        tensor = self._mace_net_def.tensors.add()
        tensor.name = output_name
        onnx_tensor = node.attrs['value']
        tensor_value = numpy_helper.to_array(onnx_tensor)
        tensor.dims.extend(list(onnx_tensor.dims))
        data_type = onnx_dtype(onnx_tensor.data_type)

        if data_type == np.float32 or data_type == np.float64:
            tensor.data_type = mace_pb2.DT_FLOAT
            tensor.float_data.extend(
                tensor_value.astype(np.float32).flat)
        elif data_type == np.int32 or data_type == np.int64:
            tensor.data_type = mace_pb2.DT_INT32
            tensor.int32_data.extend(
                tensor_value.astype(np.int32).flat)
        else:
            mace_check(False,
                       "Not supported tensor type: %s" % data_type)
        self._consts[tensor.name] = tensor

    def convert_conv2d(self, node):
        op = self.convert_general_op(node)
        self.add_stride_pad_kernel_arg(node.attrs, op)
        group_arg = op.arg.add()
        group_arg.name = MaceKeyword.mace_group_str
        if 'group' in node.attrs:
            group_val = node.attrs["group"]
        else:
            group_val = 1
        group_arg.i = group_val
        filter_shape = self._graph_shapes_dict[node.inputs[1]]
        filter_tensor = None
        try:
          filter_tensor = self._consts[node.inputs[1]]
        except KeyError:
          print('{} filter is variable tensor'.format(op.name))
        is_depthwise = False
        if group_val > 1:
            if group_val == filter_shape[0] and filter_shape[1] == 1:
                new_shape = [filter_shape[1], filter_shape[0],
                             filter_shape[2], filter_shape[3]]
                if filter_tensor != None:
                    del filter_tensor.dims[:]
                    filter_tensor.dims.extend(new_shape)
                is_depthwise = True
        if len(self._graph_shapes_dict[node.inputs[0]]) == 3:#eca_net
            #modify conv kernel_shape  3-Dim to 4-Dim
            filter_shape = self._graph_shapes_dict[node.inputs[1]]
            new_shape = [1, filter_shape[0],
                         filter_shape[1], filter_shape[2]]
            filter_tensor = self._consts[node.inputs[1]]
            del filter_tensor.dims[:]
            filter_tensor.dims.extend(new_shape)

        if is_depthwise:
            op.type = MaceOp.DepthwiseConv2d.name
        else:
            op.type = MaceOp.Conv2D.name
            mace_check(op.input[1] in self._consts,
                       "Mace does not support non-const filter convolution.")

        dilation_arg = op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        if 'dilations' in node.attrs:
            dilation_val = node.attrs["dilations"]
        else:
            dilation_val = [1, 1]
        dilation_arg.ints.extend(dilation_val)
        # for gray pic case
        if filter_shape[1] == 1 and is_depthwise == False and self._rawdata == False:
          for name in self.input_node_array:
            if name == op.input[0]:
              self.input_node_shapes_array[0][1] = 3
              filter_data = np.array(filter_tensor.float_data).reshape(filter_shape)
              [n,c,h,w] = filter_shape
              dummy = np.zeros((n,2,h,w))
              filter_data = np.concatenate((filter_data,dummy),axis=1)
              del filter_tensor.dims[:]
              filter_tensor.dims.extend(filter_data.shape)
              del filter_tensor.float_data[:]
              filter_tensor.float_data.extend(filter_data.flat)

        #add bias tensor
        if len(node.inputs) != 3:
           filter_shape = self._graph_shapes_dict[node.inputs[1]]
           bise_shape = filter_shape[0]
           bias_data = np.zeros(bise_shape)
           bias_tensor_name = op.name + '_bias'
           self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                           mace_pb2.DT_FLOAT,
                           bias_data.flat)
           op.input.extend([bias_tensor_name])

    def convert_deconv(self, node):
        op = self.convert_general_op(node)
        self.add_stride_pad_kernel_arg(node.attrs, op)


        if 'group' in node.attrs:
            group_val = node.attrs["group"]
        else:
            group_val = 1
        filter_shape = self._graph_shapes_dict[node.inputs[1]]
        filter_tensor = self._consts[node.inputs[1]]
        if group_val > 1:
          if group_val == filter_shape[0] and filter_shape[1] == 1:
            op.type = MaceOp.DepthwiseDeconv2d.name
            filter_shape = self._graph_shapes_dict[node.inputs[1]]
            filter_tensor = self._consts[node.inputs[1]]
            new_shape = [filter_shape[1], filter_shape[0],
                         filter_shape[2], filter_shape[3]]
            filter_shape = new_shape
            del filter_tensor.dims[:]
            filter_tensor.dims.extend(new_shape)
        else:
            op.type = MaceOp.Deconv2D.name
            #add by sigmastar
            #conv2d filter array : outputNum , inChannel , H , W
            #deconv filter array : inChannel , outputNum , H , W
            weight1_tensor = self._consts[node.inputs[1]]
            weight1_shape = self._graph_shapes_dict[node.inputs[1]]
            weight1_data = np.array(weight1_tensor.float_data).reshape(weight1_shape)
            weight1_data = np.transpose(weight1_data,axes=(1,0,2,3))
            del weight1_tensor.dims[:]
            weight1_tensor.dims.extend(weight1_data.shape)
            del weight1_tensor.float_data[:]
            weight1_tensor.float_data.extend(weight1_data.flat)
        group_arg = op.arg.add()
        group_arg.name = MaceKeyword.mace_group_str
        group_arg.i = group_val

        dilation_arg = op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        if 'dilations' in node.attrs:
            dilation_val = node.attrs["dilations"]
        else:
            dilation_val = [1, 1]
        dilation_arg.ints.extend(dilation_val)
        #add bias tensor
        if len(node.inputs) != 3:
           bise_shape = filter_shape[1]
           bias_data = np.zeros(bise_shape)
           bias_tensor_name = op.name + '_bias'
           self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                           mace_pb2.DT_FLOAT,
                           bias_data)
           op.input.extend([bias_tensor_name])

        mace_check(dilation_val == [1, 1],
                   "not support convtranspose with dilation != 1 yet.")

        mace_check('output_padding' not in node.attrs,
                   "not support convtranspose with output_padding yet.")
        mace_check('output_shape' not in node.attrs,
                   "not support convtranspose with output_shape yet.")
        # TODO: if output shape specified, calculate padding value
        # if 'output_padding' in node.attrs:
        #     output_padding = node.attrs['output_padding']
        #     output_padding_arg = op.arg.add()
        #     output_padding_arg.name = MaceKeyword.mace_output_padding_str
        #     output_padding_arg.ints.extend(output_padding)
        # if 'output_shape' in node.attrs:
        #     output_shape = node.attrs['output_shape']
        #     output_shape_arg = op.arg.add()
        #     output_shape_arg.name = MaceKeyword.mace_output_shape_str
        #     output_shape_arg.ints.extend(output_shape)

    def convert_depth_space(self, node):
        op = self.convert_general_op(node)
        if node.op_type == OnnxOpType.DepthToSpace.name:
            op.type = MaceOp.DepthToSpace.name
            mode = 0 # DCR = 0(default),CRD = 1
            if 'mode' in node.attrs:
              if node.attrs['mode'] == 'DCR':
                mode = 0
              else:
                mode = 1
            mode_arg = op.arg.add()
            mode_arg.name = 'mode'
            mode_arg.i = mode
        else:
            op.type = MaceOp.SpaceToDepth.name

        if 'block_size' in node.attrs:
          block_size = node.attrs['block_size']
        elif 'blocksize' in node.attrs:
          block_size = node.attrs['blocksize']
        else:
          mace_check(('block_size' in node.attrs),
                    "depth to space op should have block size attribute.")
        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_space_depth_block_size_str
        size_arg.i = block_size

    def convert_dim_range(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Slice.name

        mace_check('offset' in node.attrs,
                   "Attribute dim required!")
        mace_check('output_dim' in node.attrs,
                   "Attribute output_dim required!")
        offset = node.attrs['offset']
        starts_arg = op.arg.add()
        starts_arg.name = 'starts'
        starts_arg.ints.extend([offset])
        output_dim = node.attrs['output_dim']
        ends_arg = op.arg.add()
        ends_arg.name = 'ends'
        ends_arg.ints.extend([output_dim + offset])
        axes_arg = op.arg.add()
        axes_arg.name = 'axes'
        axes_arg.ints.extend([-1])

    def convert_dropout(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name
        #del op.output[1:]
        #del op.output_shape[1:]
        output_shape = list(op.output_shape[0].dims)
        reshape_output_tensor_name = op.name + '_output_shape'
        reshape_output_tensor_data = output_shape
        reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
        self.add_tensor(reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
        op.input.extend([reshape_output_tensor_name])

    def convert_dynamic_lstm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.DynamicLSTM.name

        self.copy_node_attr(op, node, 'prev_out_delay',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'prev_cell_delay',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'prev_out_offset',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'prev_out_dim',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'prev_cell_dim',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'bias_a',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'bias_b',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'scale',
                            AttributeType.FLOAT)
        self.copy_node_attr(op, node, 'subsample_factor',
                            AttributeType.INT, default=1)
        self.copy_node_attr(op, node, 'cell_cache_indexes',
                            AttributeType.INTS, default=[])
        self.copy_node_attr(op, node, 'out_cache_indexes',
                            AttributeType.INTS, default=[])
        self.copy_node_attr(op, node, 'forward_indexes',
                            AttributeType.INTS)

    def convert_clip(self, node):
        #  If clip's min value is zero,
        #  convert clip to activation(ReLU or ReLUX)
        #  so it can be fused into convolution.
        is_relux = False
        if len(node.inputs) > 1:
            if node.inputs[1] in self._consts:
                min_value = float(np.array(self._consts[node.inputs[1]].float_data))
                if min_value == 0.0:
                    is_relux = True
            if is_relux:
                op = self.convert_general_op(node)
                op.type = MaceOp.Activation.name
                type_arg = op.arg.add()
                type_arg.name = MaceKeyword.mace_activation_type_str
                if  (len(node.inputs)==3) and  (node.inputs[2] in self._consts):
                    max_value = float(np.array(self._consts[node.inputs[2]].float_data))
                    mace_check((max_value == 6.0),
                        "Clip OP only support max value is 6.")
                    type_arg.s = six.b(ActivationType.RELUX.name)
                    alpha_arg = op.arg.add()
                    alpha_arg.name = MaceKeyword.mace_activation_max_limit_str
                    alpha_arg.f = max_value
                    if (node.inputs[1] in self._consts) and (node.inputs[2] in self._consts):
                         op.input.remove(node.inputs[1])
                         op.input.remove(node.inputs[2])
                else:
                    type_arg.s = six.b(ActivationType.RELU.name)
            else:
               self.convert_eltwise(node)
        else:
            if 'min' in node.attrs:
                min_value = node.attrs['min']
                if min_value == 0:
                    is_relux = True
            if is_relux:
                op = self.convert_general_op(node)
                op.type = MaceOp.Activation.name

                type_arg = op.arg.add()
                type_arg.name = MaceKeyword.mace_activation_type_str
                if "max" in node.attrs:
                    max_value = node.attrs["max"]
                    mace_check((max_value == 6),
                        "not support max value is not 6.")
                    type_arg.s = six.b(ActivationType.RELUX.name)
                    alpha_arg = op.arg.add()
                    alpha_arg.name = MaceKeyword.mace_activation_max_limit_str
                    alpha_arg.f = max_value
                else:
                    type_arg.s = six.b(ActivationType.RELU.name)
            else:
                self.convert_eltwise(node)

    def convert_eltwise(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Eltwise.name
        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = self.eltwise_type[node.op_type].value
        if node.op_type == OnnxOpType.Sqrt.name:
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = 0.5
        elif node.op_type == OnnxOpType.Reciprocal.name:
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = -1
        elif node.op_type == OnnxOpType.Scale.name and 'scale' in node.attrs:
            value = node.attrs['scale']
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = value
        elif node.op_type == OnnxOpType.Clip.name:
            if len(node.inputs) > 1:
                if node.inputs[1] in self._consts:
                    self._consts[node.inputs[1]].float_data
            else:
                if 'min' in node.attrs:
                    min_value = node.attrs['min']
                else:
                    min_value = np.finfo(np.float32).min
                if 'max' in node.attrs:
                    max_value = node.attrs['max']
                else:
                    max_value = np.finfo(np.float32).max
                coeff_arg = op.arg.add()
                coeff_arg.name = MaceKeyword.mace_coeff_str
                coeff_arg.floats.extend([min_value, max_value])
        elif len(node.inputs) == 2:
            if node.inputs[1] in self._consts and \
                    node.inputs[0] not in self._consts:
                const_name = node.inputs[1]
                const_tensor = self._consts[const_name]
                if len(self._graph_shapes_dict[node.inputs[0]]) != len(const_tensor.dims):
                    tmp = 1
                    for i in range(len(const_tensor.dims)):
                        tmp *= const_tensor.dims[i]
                    const_shape = [tmp]
                    del const_tensor.dims[:]
                    const_tensor.dims.extend(const_shape)
                if len(const_tensor.dims) == 0:

                    if const_tensor.data_type == mace_pb2.DT_INT32:
                        value_data = float(const_tensor.int32_data[0])
                    elif const_tensor.data_type == mace_pb2.DT_FLOAT:
                        value_data = const_tensor.float_data[0]
                    else:
                        mace_check(False,
                                   "Does not support param's data type %s"
                                   % const_tensor.data_type)

                    del op.input[1]
                    value_shape = [1]
                    value_name = const_name
                    self.add_tensor(value_name, value_shape,
                                    const_tensor.data_type,
                                    [value_data])
                    op.input.extend([value_name])

            elif node.inputs[0] in self._consts and \
                    node.inputs[1] not in self._consts:
                const_name = node.inputs[0]
                const_tensor = self._consts[const_name]
                variable_tensor_shape = self._graph_shapes_dict[node.inputs[1]]
                if len(const_tensor.dims) == 0:

                    if const_tensor.data_type == mace_pb2.DT_INT32:
                        value_data = float(const_tensor.int32_data[0])
                    elif const_tensor.data_type == mace_pb2.DT_FLOAT:
                        value_data = const_tensor.float_data[0]
                    else:
                        mace_check(False,
                                   "Does not support param's data type %s"
                                   % const_tensor.data_type)
                    value_index_arg = op.arg.add()
                    value_index_arg.name = \
                        MaceKeyword.mace_scalar_input_index_str
                    value_index_arg.i = 0
                    del op.input[0]
                    value_shape = [1]
                    value_name = const_name
                    self.add_tensor(value_name, value_shape,
                                    const_tensor.data_type,
                                    [value_data])
                    op.input.extend([value_name])
                elif len(const_tensor.dims) != 4 and len(variable_tensor_shape) == 4:
                    #tiny_yolov2.onnx case
                    #add op
                    #   input0:3,1,1
                    #   input1:1,3,416,616
                    tmp_shape = [1,1,1,1]
                    for i in six.moves.range(len(const_tensor.dims)):
                        tmp_shape[-1-i] = const_tensor.dims[-1-i]
                    del const_tensor.dims[:]
                    const_tensor.dims.extend(list(tmp_shape))

    @staticmethod
    def copy_node_attr(op, node, attr_name, dtype=AttributeType.INT,
                       default=None):
        if attr_name in node.attrs or default is not None:
            if attr_name in node.attrs:
                value = node.attrs[attr_name]
            else:
                value = default
            new_arg = op.arg.add()
            new_arg.name = attr_name
            if dtype == AttributeType.INT:
                new_arg.i = int(value)
            elif dtype == AttributeType.FLOAT:
                new_arg.f = float(value)
            elif dtype == AttributeType.INTS:
                new_arg.ints.extend(value)
            elif dtype == AttributeType.FLOATS:
                new_arg.floats.extend(value)
            return value
        else:
            return default

    def convert_extract_pooling(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.ExtractPooling.name

        self.copy_node_attr(op, node, 'include_variance', AttributeType.INT)
        self.copy_node_attr(op, node, 'num_log_count', AttributeType.INT)
        self.copy_node_attr(op, node, 'variance_floor', AttributeType.FLOAT)
        self.copy_node_attr(op, node, 'counts', AttributeType.FLOATS)
        self.copy_node_attr(op, node, 'forward_indexes', AttributeType.INTS)

    def convert_flatten(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if 'axis' in node.attrs:
            axis_arg.i = node.attrs['axis']
        axis_arg.i = 4 + axis_arg.i if axis_arg.i < 0 else axis_arg.i

        end_axis_arg = op.arg.add()
        end_axis_arg.name = MaceKeyword.mace_end_axis_str
        end_axis_arg.i = -1

        output_shape = list(op.output_shape[0].dims)
        reshape_output_tensor_name = op.name + '_output_shape'
        reshape_output_tensor_data = output_shape
        reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
        self.add_tensor(reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
        op.input.extend([reshape_output_tensor_name])

    def convert_kaldi_batchnorm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.KaldiBatchNorm.name
        dim = self.copy_node_attr(op, node, 'dim', AttributeType.INT, -1)
        block_dim = self.copy_node_attr(op, node, 'block_dim',
                                        AttributeType.INT, -1)
        epsilon = self.copy_node_attr(op, node, 'epsilon',
                                      AttributeType.FLOAT, 1e-3)
        target_rms = self.copy_node_attr(op, node, 'target_rms',
                                         AttributeType.FLOAT, 1.0)
        test_mode = self.copy_node_attr(op, node, 'test_mode',
                                        AttributeType.INT, 0)
        mace_check(block_dim > 0 and
                   dim % block_dim == 0 and
                   epsilon > 0 and
                   target_rms > 0, "attributes invalid.")

        if test_mode > 0:
            mace_check(len(node.inputs) == 3,
                       "Kaldi's BatchNorm should have 3 inputs.")
            stats_mean = np.array(self._consts[node.inputs[1]].float_data)
            stats_var = np.array(self._consts[node.inputs[2]].float_data)
            offset_value = -1.0 * stats_mean
            scale_value = stats_var
            scale_value[scale_value < 0] = 0
            scale_value = np.power(scale_value + epsilon, -0.5) * target_rms
            offset_value = offset_value * scale_value
            scale_name = node.name + '_scale'
            offset_name = node.name + '_offset'
            self.add_tensor(scale_name, scale_value.shape,
                            mace_pb2.DT_FLOAT, scale_value)
            self.add_tensor(offset_name, offset_value.shape,
                            mace_pb2.DT_FLOAT, offset_value)
            del op.input[1:]
            op.input.extend([scale_name, offset_name])
            del op.output[1:]
            del op.output_shape[1:]

    def convert_fused_batchnorm(self, node):
        if self._isKaldi:
            self.convert_kaldi_batchnorm(node)
            return
        op = self.convert_general_op(node)
        op.type = MaceOp.BatchNorm.name

        if "epsilon" in node.attrs:
            epsilon_value = node.attrs["epsilon"]
        else:
            epsilon_value = 1e-5

        mace_check(len(node.inputs) == 5, "batch norm should have 5 inputs.")

        gamma_value = np.array(self._consts[node.inputs[1]].float_data)
        beta_value = np.array(self._consts[node.inputs[2]].float_data)
        mean_value = np.array(self._consts[node.inputs[3]].float_data)
        var_value = np.array(self._consts[node.inputs[4]].float_data)

        scale_name = node.name + '_scale'
        offset_name = node.name + '_offset'
        scale_value = ((1.0 / np.sqrt(
                    var_value + epsilon_value)) * gamma_value)
        offset_value = (-mean_value * scale_value) + beta_value
        self.add_tensor(scale_name, scale_value.shape, mace_pb2.DT_FLOAT,
                        scale_value)
        self.add_tensor(offset_name, offset_value.shape, mace_pb2.DT_FLOAT,
                        offset_value)
        del op.input[1:]
        op.input.extend([scale_name, offset_name])
        del op.output[1:]
        del op.output_shape[1:]

    def convert_gather(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Gather.name

        if 'axis' in node.attrs:
            value = node.attrs['axis']
        else:
            value = 0
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = value

    def convert_gemm(self, node):
        if self._isKaldi:
            self.convert_affine(node)
            return
        mace_check(len(node.inputs) >= 2,
                   "Gemm should have at least two inputs.")
        if 'alpha' in node.attrs:
            alpha = node.attrs['alpha']
            if alpha != 1.0 and node.inputs[1] in self._consts:
                weights = self._consts[node.inputs[1]]
                for idx in six.moves.range(self.get_tensor_len(weights)):
                    weights.float_data[idx] *= alpha
        if 'beta' in node.attrs:
            beta = node.attrs['beta']
            if beta != 1.0 and len(node.inputs) == 3 and\
                    node.inputs[2] in self._consts:
                bias = self._consts[node.inputs[2]]
                for idx in six.moves.range(self.get_tensor_len(bias)):
                    bias.float_data[idx] *= beta
        trans_a = node.attrs['transA'] if 'transA' in node.attrs else 0
        trans_b = node.attrs['transB'] if 'transB' in node.attrs else 0
        is_fc = False
        if trans_a == 0 and trans_b == 1 and\
            node.inputs[0] in self._graph_shapes_dict and\
                node.inputs[1] in self._graph_shapes_dict and \
                node.inputs[1] in self._consts:
            shape_a = self._graph_shapes_dict[node.inputs[0]]
            shape_b = self._graph_shapes_dict[node.inputs[1]]
            #add by sigmastar
            if len(shape_a) == 2 and len(shape_b) == 2:
                is_fc = True
            elif len(shape_a) == 4 and\
                    len(shape_b) == 4 and list(shape_b[2:]) == [1, 1]:
                is_fc = True
        if is_fc:
            op = self.convert_general_op(node, with_shape=False)
            op.type = MaceOp.FullyConnected.name
            for output in node.outputs:
                output_shape = op.output_shape.add()
                shape_info = self._graph_shapes_dict[output]
                mace_check(len(shape_info) in [2, 4],
                           "gemm output shape should be 2 or 4 dims.")
                if len(shape_info) == 4:
                    mace_check(list(shape_info[2:]) == [1, 1],
                               "gemm's output shape should be [*, * , 1, 1]")
                else:
                    shape_info = [shape_info[0], shape_info[1], 1, 1]
                output_shape.dims.extend(shape_info)
        else:
            op = self.convert_general_op(node)
            op.type = MaceOp.MatMul.name
            trans_a_arg = op.arg.add()
            trans_a_arg.name = MaceKeyword.mace_transpose_a_str
            trans_a_arg.i = trans_a
            trans_b_arg = op.arg.add()
            trans_b_arg.name = MaceKeyword.mace_transpose_b_str
            trans_b_arg.i = trans_b

    def convert_identity(self, node):
        self.remove_node(node)
        #op = self.convert_general_op(node)
        #op.type = MaceOp.Identity.name

    def convert_ifdefined(self, node):
        op = self.convert_general_op(node)
        if 'offset' in node.attrs:
            offset = node.attrs['offset']
        else:
            offset = 0
        mace_check(offset <= 0, "IfDefined's offset should be <= 0.")
        if offset == 0:
            op.type = MaceOp.Identity.name
        else:
            op.type = MaceOp.IfDefined.name
            self.copy_node_attr(op, node, 'forward_indexes',
                                AttributeType.INTS)
            self.copy_node_attr(op, node, 'cache_forward_indexes',
                                AttributeType.INTS)

    def convert_imagescaler(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.BatchNorm.name

        scale = node.attrs['scale']
        bias_value = np.array(node.attrs['bias'])
        scale_value = scale * np.ones_like(bias_value)

        scale_name = node.name + "_scale"
        bias_name = node.name + "_bias"
        self.add_tensor(scale_name, scale_value.shape,
                        mace_pb2.DT_FLOAT, scale_value)
        self.add_tensor(bias_name, bias_value.shape,
                        mace_pb2.DT_FLOAT, bias_value)
        op.input.extend([scale_name, bias_name])

    def convert_lstm(self, node):
        op = self.convert_general_op(node)
        op.type = 'LSTM'
        weight1_data_ori = self._consts[node.inputs[1]].float_data
        weight1_shape = self._graph_shapes_dict[node.inputs[1]]
        weight1_data = np.array(weight1_data_ori).reshape(weight1_shape)
        weight2_data_ori = self._consts[node.inputs[2]].float_data
        weight2_shape = self._graph_shapes_dict[node.inputs[2]]
        weight2_data = np.array(weight2_data_ori).reshape(weight2_shape)
        weight_data = np.concatenate((weight1_data, weight2_data), axis=-1)
        weight_data.shape = [weight_data.shape[-2],weight_data.shape[-1],1,1]

        bias = self._consts[node.inputs[3]].float_data
        bias_shape = self._graph_shapes_dict[node.inputs[3]]
        bias_ori_data = np.array(bias).reshape(bias_shape)
        split_num = weight1_shape[-2]
        hidden_size = split_num//4
        if bias_shape[-1] > split_num:
            bias1_data = bias_ori_data[0][:split_num]
            bias2_data = bias_ori_data[0][split_num:]
            bias_data = np.add(bias1_data,bias2_data)
        else:
            bias_data = bias_ori_data
        np.savez("./weight_biase_data", weight = weight_data,bias = bias_data)

    def convert_lstm_nonlinear(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.LstmNonlinear.name

    def convert_matmul(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.MatMul.name

    def convert_nop(self, node):
        pass

    def convert_normalize(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.BatchNorm.name

    def convert_pad(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Pad.name
        if 'mode' in node.attrs:
            mace_check(node.attrs['mode'] == 'constant', 'Pad only support `constant` mode!')
        if 'pads' in node.attrs:
            paddings_arg = op.arg.add()
            paddings_arg.name = MaceKeyword.mace_paddings_str
            paddings_value = node.attrs['pads']
            paddings_arg.ints.extend(paddings_value)
            if len(op.input) == 1:
                paddings_tensor_name = op.name + '_' + paddings_arg.name
                paddings_tensor_name_data = paddings_value
                paddings_tensor_name_shape = [len(paddings_tensor_name_data)]
                self.add_tensor(paddings_tensor_name, paddings_tensor_name_shape,
                                mace_pb2.DT_INT32, paddings_tensor_name_data)
                op.input.extend([paddings_tensor_name])
        if 'value' in node.attrs:
            mace_check(node.attrs['value'] == 0, 'Pad value only support padding 0!')

    def convert_pad_context(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.PadContext.name
        if 'left_context' in node.attrs:
            left_context_arg = op.arg.add()
            left_context_arg.name = 'left_context'
            left_context_arg.i = node.attrs['left_context']
        if 'right_context' in node.attrs:
            right_context_arg = op.arg.add()
            right_context_arg.name = 'right_context'
            right_context_arg.i = node.attrs['right_context']

    def convert_pnorm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.PNorm.name
        if 'output_dim' in node.attrs:
            output_dim_arg = op.arg.add()
            output_dim_arg.name = 'output_dim'
            output_dim_arg.i = node.attrs['output_dim']
        if 'p' in node.attrs:
            p_value = node.attrs['p']
            mace_check((p_value >= 0) and (p_value <= 2),
                       "PNorm only supports p = 0, 1, 2")
            p_arg = op.arg.add()
            p_arg.name = 'p'
            p_arg.i = p_value

    def convert_pooling(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Pooling.name
        self.add_stride_pad_kernel_arg(node.attrs, op)
        pooling_type_arg = op.arg.add()
        pooling_type_arg.name = MaceKeyword.mace_pooling_type_str
        pooling_type_arg.i = self.pooling_type_mode[node.op_type].value

        round_mode_arg = op.arg.add()
        round_mode_arg.name = MaceKeyword.mace_round_mode_str
        round_mode_arg.i = RoundMode.FLOOR.value

    def convert_reduce(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reduce.name

        reduce_type_arg = op.arg.add()
        reduce_type_arg.name = MaceKeyword.mace_reduce_type_str
        reduce_type_arg.i = self.reduce_type[node.op_type].value

        if node.op_type in [OnnxOpType.GlobalAveragePool.name,
                            OnnxOpType.GlobalMaxPool.name]:
            reduce_dims = [2, 3]
            keep_dims = 1
        else:
            if 'axes' in node.attrs:
                reduce_dims = node.attrs['axes']
                if len(reduce_dims) == 2:
                   mace_check(reduce_dims == [2, 3],
                        "only support multi axes [2,3]")
                   op.type = "Reduce_mean"
                   #add axis tensor
                   axis_tensor_name = op.name + '_axis'
                   axis_shape = [2]
                   axis_data = [1,2]
                   self.add_tensor(axis_tensor_name, axis_shape,
                            mace_pb2.DT_INT32,axis_data)
                   op.input.extend([axis_tensor_name])
            else:
                reduce_dims = []
            if 'keepdims' in node.attrs:
                keep_dims = node.attrs['keepdims']
            else:
                keep_dims = 1
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.ints.extend(reduce_dims)

        keep_dims_arg = op.arg.add()
        keep_dims_arg.name = MaceKeyword.mace_keepdims_str
        keep_dims_arg.i = keep_dims
    '''
    def convert_reduce_mean(self, node):
        if self._mace_net_def.op[-1].type == 'Reduce_mean':
           op = self._mace_net_def.op[-1]
           op.output.pop(0)#移除上一个reduce_mean输出
           op.output.append(node.outputs[0])#将上一个节点替换为当前节点的输出

           op.output_shape.pop(0)
           output_shape = op.output_shape.add()
           shape_info = self._graph_shapes_dict[node.outputs[0]]
           output_shape.dims.extend(shape_info)

        else:
            op = self.convert_general_op(node)
            op.type = "Reduce_mean"
            reduce_type_arg = op.arg.add()
            reduce_type_arg.name = MaceKeyword.mace_reduce_type_str
            reduce_type_arg.i = self.reduce_type[node.op_type].value
            if node.op_type in [OnnxOpType.GlobalAveragePool.name,
                                OnnxOpType.GlobalMaxPool.name]:
                reduce_dims = [2, 3]
                keep_dims = 1
            else:
                if 'axes' in node.attrs:
                    reduce_dims = node.attrs['axes']
                else:
                    reduce_dims = []
                if 'keepdims' in node.attrs:
                    keep_dims = node.attrs['keepdims']
                else:
                    keep_dims = 1
            axis_arg = op.arg.add()
            axis_arg.name = MaceKeyword.mace_axis_str
            axis_arg.ints.extend(reduce_dims)
            keep_dims_arg = op.arg.add()
            keep_dims_arg.name = MaceKeyword.mace_keepdims_str
            keep_dims_arg.i = keep_dims
            axis_shape = [2]
            axis_data = [1,2]
            axis_tensor_name = op.name + '_axis'
            self.add_tensor(axis_tensor_name, axis_shape,
                            mace_pb2.DT_INT32,
                            axis_data)
            op.input.extend([axis_tensor_name])
    '''
    def convert_replaceindex(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.ReplaceIndex.name
        self.copy_node_attr(op, node, 'forward_indexes',
                            AttributeType.INTS)

    def convert_reshape(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name

    def convert_shape(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Shape.name
        op.output_type.extend([mace_pb2.DT_INT32])

    def convert_slice(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Slice.name
        if len(node.attrs) > 0:
            mace_check(len(op.input) == 1, "ONNX Slice(Ver 9) need 1 input")
            mace_check('starts' in node.attrs, "Attribute starts required!")
            mace_check('ends' in node.attrs, "Attribute ends required!")
            for attr in node.attrs:
                mace_check(attr in ['axes', 'ends', 'starts'],
                           "onnx_converter didn't support %s attribute yet!" % attr)
            starts_arg = op.arg.add()
            starts_arg.name = op.name + '_starts'
            starts_arg.i = node.attrs['starts']

            ends_arg = op.arg.add()
            ends_arg.name = op.name + '_ends'
            ends_arg.i = node.attrs['ends']

            axis_arg = op.arg.add()
            axis_arg.name = op.name + '_axis'
            axis_arg.i = node.attrs.get('axes', -1)

            steps_arg = op.arg.add()
            steps_arg.name = op.name + '_steps'
            steps_arg.i = node.attrs.get('steps', 1)

        else:
            mace_check(len(op.input) >= 3,
            "ONNX Slice(Ver 10 or higher) need >= 3 inputs, See `https://github.com/onnx/onnx/blob/master/docs/Operators.md` for details")

            start_tensor = self._consts[node.inputs[1]].int32_data[0]
            starts_arg = op.arg.add()
            starts_arg.name = op.name + '_starts'
            starts_arg.i = start_tensor

            end_tensor = self._consts[node.inputs[2]].int32_data[0]
            ends_arg = op.arg.add()
            ends_arg.name = op.name + '_ends'
            ends_arg.i = end_tensor

            try:
                axis_tensor = self._consts[node.inputs[3]].int32_data[0]
            except IndexError:
                axis_tensor = -1
            axis_arg = op.arg.add()
            axis_arg.name = op.name + '_axis'
            axis_arg.i = axis_tensor

            try:
                steps_tensor = self._consts[node.inputs[4]].int32_data[0]
            except IndexError:
                steps_tensor = 1
            steps_arg = op.arg.add()
            steps_arg.name = op.name + '_steps'
            steps_arg.i = steps_tensor


    def convert_softmax(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Softmax.name
        if node.op_type == OnnxOpType.LogSoftmax.name:
            use_log_arg = op.arg.add()
            use_log_arg.name = 'use_log'
            use_log_arg.i = 1

    def convert_lpnormalization(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.LpNorm.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = node.attrs.get('axis', -1)

        p_arg = op.arg.add()
        p_arg.name = MaceKeyword.mace_p_str
        p_arg.i = node.attrs.get('p', 2)

    def convert_splice(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Splice.name
        if 'context' in node.attrs:
            context = node.attrs['context']
        else:
            context = [0]
        context_arg = op.arg.add()
        context_arg.name = 'context'
        context_arg.ints.extend(context)
        if 'const_component_dim' in node.attrs:
            const_dim = node.attrs['const_component_dim']
            const_dim_arg = op.arg.add()
            const_dim_arg.name = 'const_component_dim'
            const_dim_arg.i = const_dim
            self.copy_node_attr(op, node,
                                'forward_const_indexes',
                                AttributeType.INTS)

        self.copy_node_attr(op, node, 'subsample_factor',
                            AttributeType.INT, default=1)
        self.copy_node_attr(op, node, 'forward_indexes',
                            AttributeType.INTS)

    def convert_split(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Split.name
        if 'axis' in node.attrs:
            value = node.attrs['axis']
        else:
            value = 0
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = value
        if 'split' in node.attrs:
          value_1 = node.attrs['split']
          split_arg = op.arg.add()
          split_arg.name = 'slice_point'
          split_arg.ints.extend(value_1)

    def convert_squeeze(self, node):
        axis_value = node.attrs['axes']
        if node.inputs[0] in self._consts:
            tensor = self._consts[node.inputs[0]]
            shape = tensor.dims
            new_shape = self.squeeze_shape(shape, axis_value)
            del tensor.dims[:]
            tensor.dims.extend(new_shape)
            self.remove_node(node)
        else:
            op = self.convert_general_op(node)
            op.type = MaceOp.Reshape.name

            output_shape = list(op.output_shape[0].dims)
            reshape_output_tensor_name = op.name + '_output_shape'
            reshape_output_tensor_data = output_shape
            reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
            self.add_tensor(reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
            op.input.extend([reshape_output_tensor_name])

            '''
            axis_arg = op.arg.add()
            axis_arg.name = MaceKeyword.mace_axis_str
            if 'axes' in node.attrs:
                axis_value = node.attrs['axes']
            else:
                axis_value = []
            axis_arg.ints.extend(axis_value)
            '''
    def convert_unsqueeze(self, node):
        mace_check('axes' in node.attrs,
                   "Unsqueeze op should have 'axes' attribute.")
        axis_value = node.attrs['axes']
        if node.inputs[0] in self._consts:
            #tensor = self._consts[node.inputs[0]]
            #shape = tensor.dims
            #new_shape = self.unsqueeze_shape(shape, axis_value)
            #del tensor.dims[:]
            #tensor.dims.extend(shape)
            self.remove_node(node)
        else:
            #mace_check(not(True),
                           #"do not surpport yet!")
            op = self.convert_general_op(node)
            op.type = MaceOp.Reshape.name
            output_shape = list(op.output_shape[0].dims)
            reshape_output_tensor_name = op.name + '_output_shape'
            reshape_output_tensor_data = output_shape
            reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
            self.add_tensor(reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
            op.input.extend([reshape_output_tensor_name])

    def convert_subsample(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Subsample.name
        self.copy_node_attr(op, node, 'forward_indexes',
                            AttributeType.INTS)

    def convert_sum_group(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.SumGroup.name

    def convert_target_rms_norm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.TargetRMSNorm.name

        self.copy_node_attr(op, node, 'target_rms',
                            AttributeType.FLOAT)
        self.copy_node_attr(op, node, 'add_log_stddev',
                            AttributeType.INT, default=0)
        self.copy_node_attr(op, node, 'block_dim',
                            AttributeType.INT, default=0)

    def convert_transpose(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Transpose.name
        if 'perm' in node.attrs:
            perm = node.attrs['perm']
            ordered_perm = np.sort(perm)
            if np.array_equal(perm, ordered_perm):
                op.type = MaceOp.Identity.name
                del op.input[1:]
            else:
                dims_arg = op.arg.add()
                dims_arg.name = MaceKeyword.mace_dims_str
                dims_arg.ints.extend(perm)

    def convert_tile(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Tile.name
        tensor = self._consts[node.inputs[1]]
        shape = tensor.dims[0]
        if shape == 4:
            repeat_data_ori = tensor.int32_data
            repeat_data_new = [repeat_data_ori[0],repeat_data_ori[2],repeat_data_ori[3],repeat_data_ori[1]]
            del tensor.int32_data[:]
            tensor.int32_data.extend(repeat_data_new)

    def convert_timeoffset(self, node):
        op = self.convert_general_op(node)
        mace_check('offset' in node.attrs,
                   'Offset attribute required in Offset Node.')
        offset = node.attrs['offset']
        if offset == 0:
            op.type = MaceOp.Identity.name
        else:
            op.type = MaceOp.TimeOffset.name

        chunk_size = node.attrs['chunk_size']
        chunk_size_arg = op.arg.add()
        chunk_size_arg.name = 'chunk_size'
        chunk_size_arg.i = chunk_size

        offset_arg = op.arg.add()
        offset_arg.name = 'offset'
        offset_arg.i = offset

    def convert_upsample(self, node):
        op = self.convert_general_op(node)
        del op.input[1:]  # cut all unnecessary inputs (onnx>=1.5)
        input_size = self._graph_shapes_dict[op.input[0]]
        output_size = self._graph_shapes_dict[op.output[0]]
        output_size = np.array(output_size[-2:]).astype(np.int32)
        if node.attrs['mode'] == 'nearest':
            #op.type = MaceOp.ResizeNearestNeighbor.name
            op.type = 'Upsample'
            size_tensor_name = op.name + ":size"
            self.add_tensor(size_tensor_name, output_size.shape,
                            mace_pb2.DT_INT32, output_size)
            op.input.append(size_tensor_name)
        else:
            op.type = MaceOp.ResizeBilinear.name
            size_tensor_name = op.name + ":size"
            self.add_tensor(size_tensor_name, output_size.shape,
                            mace_pb2.DT_INT32, output_size)
            op.input.append(size_tensor_name)
            '''
            size_arg = op.arg.add()
            size_arg.name = MaceKeyword.mace_resize_size_str
            size_arg.ints.extend(output_size.tolist())
            '''

        align_corners_arg = op.arg.add()
        align_corners_arg.name = MaceKeyword.mace_align_corners_str
        align_corners_arg.i = node.attrs.get('align_corners', 0)

        scale_w = output_size[-1]//input_size[-1]
        scale_h = output_size[-2]//input_size[-2]
        mace_check(scale_w == scale_h,"Upsample OP only support same scale on H W")
        scale_arg = op.arg.add()
        scale_arg.name = "scale"
        scale_arg.i = scale_w

    def convert_resize(self,node):
        #
        #I6E  nearest => tile only upsample
        #     bilinear only upsample
        #M6   nearest
        #     bilinear
        op = self.convert_general_op(node)
        input_size = self._graph_shapes_dict[op.input[0]]
        output_size = self._graph_shapes_dict[op.output[0]]
        output_size = np.array(output_size[-2:]).astype(np.int32)
        #node.attrs
        zoomOut =  1 if output_size[-1] >= input_size[-1] else 0
        if getIPUVersion() == 'I6E':
          scale_w = output_size[-1]//input_size[-1]
          scale_h = output_size[-2]//input_size[-2]
          mace_check(scale_w == scale_h,"Upsample OP only support same scale on H W")
          mace_check(scale_w > 0,"Resize OP only support UpSample case")
        if node.attrs['mode'] == 'nearest':
          if getIPUVersion() == 'I6E':
            op.type = "Upsample"
            #set sigmastar's op "Upsample" input
            #add for adapter onnx version 10
            scale_arg = op.arg.add()
            scale_arg.name = 'scale'
            scale_arg.i = scale_w
          else:
            op.type = MaceOp.ResizeNearestNeighbor.name
        elif node.attrs['mode'] == 'linear':
            op.type = MaceOp.ResizeBilinear.name
        else:
            mace_check(False,"Do not support cubic mode")
        size_tensor_name = op.name + "_size"
        self.add_tensor(size_tensor_name, output_size.shape,
                        mace_pb2.DT_INT32, output_size)
        del op.input[1:]
        op.input.append(size_tensor_name)

        #add for adapter onnx version > 10
        if op.type != "Upsample":
            try:
              if  node.attrs['coordinate_transformation_mode'] != "align_corners" and node.attrs['coordinate_transformation_mode'] != "asymmetric":
                mace_check(False,"Resize OP only support 'coordinate_transformation_mode' : align_corners and asymmetric")
            except KeyError:
                return #default para
            align_corners_arg = op.arg.add()
            align_corners_arg.name = "align_corners"
            if node.attrs['coordinate_transformation_mode'] == "align_corners":
                align_corners_arg.i = 1
            if node.attrs['coordinate_transformation_mode'] == "asymmetric":
                align_corners_arg.i = 0
        #fixed redmind 13932
        if op.type == MaceOp.ResizeNearestNeighbor.name:
            if zoomOut:
              if align_corners_arg.i == 0:
                  try:
                    if node.attrs['nearest_mode'] != "round_prefer_floor" and node.attrs['nearest_mode'] != "floor":
                        mace_check(False," 'zoomOut model : coordinate_transformation_mode' is asymmetric  'nearest_mode' only support round_prefer_floor(Default) and floor")
                  except KeyError:
                      pass
              if align_corners_arg.i == 1:
                  try:
                    if node.attrs['nearest_mode'] != "round_prefer_floor" and node.attrs['nearest_mode'] != "round_prefer_ceil ":
                        mace_check(False," 'zoomOut model : coordinate_transformation_mode' is align_corners  'nearest_mode' only support round_prefer_floor(Default) and round_prefer_ceil ")
                  except KeyError:
                      pass
            else:
              if align_corners_arg.i == 0:
                  try:
                    if node.attrs['nearest_mode'] == "floor" :
                        mace_check(False," 'zoomIn model : coordinate_transformation_mode' is asymmetric  'nearest_mode' is not support floor ")
                  except KeyError:
                      pass

    def convert_expand(self,node):
        op = self.convert_general_op(node)
        op.type = 'Expand'

    def convert_exp(self,node):
        op = self.convert_general_op(node)
        op.type = 'Exp'

