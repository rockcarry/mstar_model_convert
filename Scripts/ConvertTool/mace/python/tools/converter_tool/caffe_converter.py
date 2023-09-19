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
import google.protobuf.text_format
import pdb
import sys
from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool import shape_inference
from mace.python.tools.converter_tool.base_converter import PoolingType
from mace.python.tools.converter_tool.base_converter import ActivationType
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import FrameworkType
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import FilterFormat
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.convert_util import mace_check
from mace.python.tools.convert_util import getIPUVersion

from mace.python.tools.convert_util import *
from third_party.caffe import caffe_pb2
from third_party.crypto.vendor_crypto import *
from io import BytesIO

caffe_group_str = 'group'
caffe_kernel_h_str = 'kernel_h'
caffe_kernel_w_str = 'kernel_w'
caffe_stride_h_str = 'stride_h'
caffe_stride_w_str = 'stride_w'
caffe_pad_h_str = 'pad_h'
caffe_pad_w_str = 'pad_w'


class CaffeOperator(object):
    """CaffeOperator merges and provides both layer and weights information.
    Layer records caffe layer proto, while blobs records the weight data in
    format of numpy ndarray.
    """

    def __init__(self):
        self._layer = None
        self._blobs = None

    @property
    def name(self):
        return self._layer.name

    @property
    def type(self):
        return self._layer.type

    @property
    def layer(self):
        return self._layer

    @property
    def blobs(self):
        return self._blobs

    @layer.setter
    def layer(self, layer):
        self._layer = layer

    @blobs.setter
    def blobs(self, blobs):
        self._blobs = [self.blob_to_nparray(blob) for blob in blobs]

    def get_blob(self, index):
        mace_check(index < len(self._blobs), "blob out of index")
        return self._blobs[index]

    @staticmethod
    def blob_to_nparray(blob):
        if blob.num != 0:
            return (np.asarray(blob.data, dtype=np.float32).reshape(
                (blob.num, blob.channels, blob.height, blob.width)))
        else:
            return np.asarray(blob.data, dtype=np.float32).reshape(
                blob.shape.dim)


class CaffeNet(object):
    """CaffeNet contains caffe operations. Output of each layer has unique
    name as we replace duplicated output name with unique one, while keep
    mace input/output name which user specifies unchanged."""

    def __init__(self,option):
        self._ops = {}
        self._consumers = {}
        # for in-place op, its input name is the same with output name,
        # so we change the output name to an alias
        self._alias_op_output_name = {}
        self._used_op_output_name = set()
        self._option = option
        self.output_node_array = []
        for output_node in self._option.output_nodes.values():
            self.output_node_array.append(output_node.name)

    @property
    def ops(self):
        return self._ops.values()

    def get_op(self, op_name):
        return self._ops.get(op_name, None)

    def get_consumers(self, tensor_name):
        return self._consumers.get(tensor_name, [])

    def add_layer(self, layer):
        op = CaffeOperator()
        op.layer = layer
        self._ops[layer.name] = op
        # change op output name if it is an in-place op
        layer.bottom[:] = [self._alias_op_output_name.get(layer_input,
                                                          layer_input) for
                           layer_input in layer.bottom][:]
        for i in six.moves.range(len(layer.top)):
            old_name = layer.top[i]
            if layer.type == 'Input' or layer.type == 'DummyData':
                new_name = old_name
            else:
                idx = 0
                if old_name in self.output_node_array:
                    new_name = old_name + "#" + str(idx)
                    idx += 1
                else:
                    new_name = old_name
                postfix = "_xx"
                while new_name in self._used_op_output_name:
                    new_name = new_name + postfix
            layer.top[i] = new_name
            self._alias_op_output_name[old_name] = new_name
            self._used_op_output_name.update([new_name])
        for input_tensor in layer.bottom:
            if input_tensor not in self._consumers:
                self._consumers[input_tensor] = []
            self._consumers[input_tensor].append(op)

    def add_blob(self, weight):
        if weight.name in self._ops:
            op = self._ops[weight.name]
            op.blobs = list(weight.blobs)


class CaffeConverter(base_converter.ConverterInterface):
    """A class for convert caffe model to mace model."""

    pooling_type_mode = {
        caffe_pb2.PoolingParameter.AVE: PoolingType.AVG,
        caffe_pb2.PoolingParameter.MAX: PoolingType.MAX
    }
    eltwise_type = {
        caffe_pb2.EltwiseParameter.PROD: EltwiseType.PROD,
        caffe_pb2.EltwiseParameter.SUM: EltwiseType.SUM,
        caffe_pb2.EltwiseParameter.MAX: EltwiseType.MAX,
    }
    activation_type = {
        'ReLU': ActivationType.RELU,
        'PReLU': ActivationType.PRELU,
        'TanH': ActivationType.TANH,
        'Sigmoid': ActivationType.SIGMOID,
        'ReLU6': ActivationType.RELU6,
    }

    def __init__(self, option, src_model_file, src_weight_file, input_name_format_map,is_decrypt=False):
        self._op_converters = {
            'Input': self.convert_input,
            'DummyData': self.convert_input,
            'Convolution': self.convert_conv2d,
            'Deconvolution': self.convert_deconv2d,
            'Eltwise': self.convert_elementwise,
            'Add': self.convert_add,
            'ReLU': self.convert_activation,
            'ReLU6': self.convert_activation,
            'TanH': self.convert_activation,
            'Sigmoid': self.convert_activation,
            'PReLU': self.convert_activation,
            'Pooling': self.convert_pooling,
            'Concat': self.convert_concat,
            'Slice': self.convert_slice,
            'Split': self.convert_split,
            'Softmax': self.convert_softmax,
            'InnerProduct': self.convert_fully_connected,
            'Interp': self.convert_interp,
            'BatchNorm': self.convert_folded_batchnorm,
            'Crop': self.convert_crop,
            'Scale': self.convert_scale,
            'ShuffleChannel': self.convert_channel_shuffle,
            'Permute': self.convert_permute,
            'Flatten': self.convert_flatten,
            'PriorBox': self.convert_prior_box,
            'Reshape': self.convert_reshape,
            #add by sigmastar
            'ArgMax':self.convert_ArgMax,
            'Axpy':self.convert_Axpy,
            'CReLU':self.convert_CReLU,
            'ConvolutionDepthwise': self.convert_ConvolutionDepthwise,
            'Clip':self.convert_Clip,
            'Dropout':self.convert_Dropout,
            'Upsample': self.convert_Upsample,
            'Reorg':self.convert_Reorg,
            'Reverse':self.convert_Reverse,
            'LSTM':self.convert_LSTM,
            'Threshold':self.convert_Threshold,
            'Tile':self.convert_Tile,
            'ContinuationIndicator':self.convert_ContinuationIndicator,
            'ROIPooling':self.convert_ROIPooling,
            'Power':self.convert_Power,
            'SGS_SSD_Postprocess':self.conver_SGS_CAFFE_SSD_Postprocess,
            'SGS_YoloV2_Postprocess':self.conver_SGS_YoloV2_Postprocess,
            'SGS_YoloV3_Postprocess':self.conver_SGS_YoloV3_Postprocess,
            'SGS_LanceNet_Postprocess':self.conver_SGS_LanceNet_Postprocess,
            'SGS_FDA_Postprocess':self.conver_SGS_FDA_Postprocess,
            'Custom': self.convert_Custom,
            'PriorBox_RFC': self.convert_prior_box_RFC,
            #'Normalize': self.convert_Normalize,
        }
        self._option = option
        self._mace_net_def = mace_pb2.NetDef()
        ConverterUtil.set_filter_format(self._mace_net_def, FilterFormat.OIHW)
        self._caffe_net = CaffeNet(option)
        self._caffe_layers = caffe_pb2.NetParameter()
        caffe_weights = caffe_pb2.NetParameter()
        self.input_node_array = []
        self._input_name_shape_map = {}
        self.gray_model = 0
        self._input_name_format_map = input_name_format_map
        self._lstm_num = 0
        for input_node in self._option.input_nodes.values():
            self.input_node_array.append(input_node.name)
        # parse prototxt
        if isinstance(src_model_file,str):
            f =  open(src_model_file, 'r')
            google.protobuf.text_format.Merge(
                str(f.read()), self._caffe_layers)
        else:
            if is_decrypt:
                decrypt_model_buf = decrypt(src_model_file)
                model_BytesIO = decrypt_model_buf.read()
                google.protobuf.text_format.Merge(
                    model_BytesIO, self._caffe_layers)
            else:
                google.protobuf.text_format.Merge(
                    src_model_file.read(), self._caffe_layers)

        self.filter_test_layers(self._caffe_layers)
        if len(self._caffe_layers.input_dim) != 0:
          tmp = self._caffe_layers.input_dim
          for i in six.moves.range(len(self._caffe_layers.input)):
            tmp1 = tmp[4*i:4*(1+i)]
            if len(tmp1) == 4:
              mace_check(tmp1[0] == 1, "SGS only support N = 1 for 4 dims inputs")
              if tmp1[1] == 1:
               self.gray_model +=1
            self._input_name_shape_map[self.input_node_array[i]]=tmp1

        elif len(self._caffe_layers.input_shape) != 0:
            tmp = self._caffe_layers.input_shape
            for i in six.moves.range(len(tmp)):
              dims = self._caffe_layers.input_shape[i].dim
              if len(dims) == 4:
                mace_check(dims[0] == 1, "SGS only support N = 1 for 4 dims inputs")
                if dims[1] == 1:
                  self.gray_model += 1
              self._input_name_shape_map[self.input_node_array[i]]=dims
        for layer in self._caffe_layers.layer:
          #for name in self._change_array:
            #if layer.name == "conv1_1":
              self._caffe_net.add_layer(layer)

        # parse model weight
        if type(src_weight_file) is str:
            f = open(src_weight_file, 'rb')
            caffe_weights.ParseFromString(f.read())
        else:
            weight_BytesIO = src_weight_file.read()
            caffe_weights.ParseFromString(weight_BytesIO)

        #add for compitable DEPRECATED parm layers
        if len(caffe_weights.layer) == 0:
            layers = caffe_weights.layers
        else:
            layers = caffe_weights.layer
        #self.filter_test_layers(caffe_weights)
        '''dump to file begin'''
        '''
        import sys
        f = open('caffemodel.txt', 'w')
        f.write(str(caffe_weights.layer))
        f.write(str(caffe_weights))
        f.close()
        '''
        '''dump to file end'''
        for weight in layers:
          #for name in self._change_array:
            self._caffe_net.add_blob(weight)

        self._skip_ops = []

    def run(self):
        self.convert_ops()
        for i,input_node in enumerate(self._option.input_nodes.values()):
            input_node.shape = self._input_name_shape_map[input_node.name]
        shape_inferer = shape_inference.ShapeInference(
            self._mace_net_def,
            self._option.input_nodes.values())
        shape_inferer.run()
        self.replace_output_tensor_name()
        return self._mace_net_def,self._input_name_shape_map

    @staticmethod
    def replace_input_name(ops, src_name, dst_name):
        for op in ops:
            for i in six.moves.range(len(op.input)):
                if op.input[i] == src_name:
                    op.input[i] = dst_name

    def replace_output_tensor_name(self):
        consumers = {}
        for op in self._mace_net_def.op:
            for input_name in op.input:
                if input_name not in consumers:
                    consumers[input_name] = []
                consumers[input_name].append(op)

        # replace the last op with same prefix name with the original top name
        ops = [op for op in self._mace_net_def.op]
        ops.reverse()
        visited = set()
        for op in ops:
            for i in six.moves.range(len(op.output)):
                original_output_name = op.output[i].split('#')[0]
                if original_output_name not in visited and\
                        original_output_name not in self._option.input_nodes:
                    self.replace_input_name(
                        consumers.get(op.output[i], []),
                        op.output[i],
                        original_output_name)
                    op.output[i] = original_output_name
                    visited.update([original_output_name])

        # if user set op name as output node, replace it with op name
        for op in self._mace_net_def.op:
            if op.name in self._option.output_nodes and op.name not in visited:
                if len(op.output) > 0:
                    self.replace_input_name(
                        consumers.get(op.output[0], []),
                        op.output,
                        op.name)
                    op.output[0] = op.name

    @staticmethod
    def filter_test_layers(layers):
        phase_map = {0: 'train', 1: 'test'}
        while True:
            changed = False
            for layer in layers.layer:
                phase = 'test'
                if len(layer.include):
                    phase = phase_map[layer.include[0].phase]
                if len(layer.exclude):
                    phase = phase_map[layer.exclude[0].phase]
                if phase != 'test':
                    print("Remove layer %s (%s)" % (layer.name, layer.type))
                    layers.layer.remove(layer)
                    changed = True
                    break
            if not changed:
                break

    @staticmethod
    def add_stride_pad_kernel_arg(param, op_def):
        try:
            if len(param.stride) > 1 or len(param.kernel_size) > 1 or len(
                    param.pad) > 1:
                raise Exception(
                    'Mace does not support multiple stride/kernel_size/pad')
            stride = [param.stride[0],
                      param.stride[0]] if len(param.stride) else [1, 1]
            pad = [param.pad[0],param.pad[0],
                   param.pad[0],param.pad[0]] if len(param.pad) else [0, 0, 0, 0]#l,r,t.b
            kernel = [param.kernel_size[0], param.kernel_size[0]] if len(
                param.kernel_size) else [0, 0]
        except TypeError:
            stride = [param.stride, param.stride]
            if param.pad != None:
              pad = [param.pad, param.pad, param.pad, param.pad] # l,r,t,b
            elif param.pad_w != None:
              pad = [param.pad_w,param.pad_w,param.pad_h, param.pad_h] # l,r,t,b
            else:
              pad = [0,0,0,0]
            if param.kernel_size != None:
              kernel = [param.kernel_size, param.kernel_size]

        if param.HasField(caffe_stride_h_str) or param.HasField(
                caffe_stride_w_str):
            stride = [param.stride_h, param.stride_w]
        if param.HasField(caffe_pad_h_str) or param.HasField(caffe_pad_w_str):
            pad = [param.pad_w,param.pad_w,param.pad_h, param.pad_h]
        if param.HasField(caffe_kernel_h_str) or param.HasField(
                caffe_kernel_w_str):
            kernel = [param.kernel_h, param.kernel_w]

        strides_arg = op_def.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(stride)
        padding_arg = op_def.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_values_str
        padding_arg.ints.extend(pad)
        kernels_arg = op_def.arg.add()
        kernels_arg.name = MaceKeyword.mace_kernel_str
        kernels_arg.ints.extend(kernel)
        if op_def.type == MaceOp.Pooling.name:
            # ceil_mode is default
            ceil_mode_arg = op_def.arg.add()
            ceil_mode_arg.name = 'ceil_mode'
            ceil_mode_arg.i = 1
            if param.HasField(caffe_kernel_h_str) or param.HasField(
                    caffe_kernel_w_str):
                kernel = [param.kernel_h, param.kernel_w]
            kernels_arg = op_def.arg.add()
            kernels_arg.name = MaceKeyword.mace_kernel_str
            kernels_arg.ints.extend(kernel)
            if param.HasField('global_pooling'):
              if param.global_pooling == True:
                global_pooling_arg = op_def.arg.add()
                global_pooling_arg.name = MaceKeyword.mace_global_pooling_str
                global_pooling_arg.i = 1
            if param.HasField('ceil_mode'):
              if param.ceil_mode == False:
                ceil_mode_arg.i = 0
            if param.HasField('round_mode'):
                if param.round_mode == 1:
                  ceil_mode_arg.i = 0
        if op_def.type == MaceOp.Conv2D.name:
            #add group arg
            group_arg = op_def.arg.add()
            group_arg.name = 'group'
            group_arg.i = 1
            if param.HasField('group'):
                group_arg.i = param.group

    def convert_ops(self):
        layer_names = set()
        for layer in self._caffe_layers.layer:
        #for test
         #for name in self._change_array:
           #if layer.name == name:
            caffe_op = self._caffe_net.get_op(layer.name)
            if caffe_op not in self._skip_ops:
                mace_check(layer.name not in layer_names,
                           "There is duplicate layer name '%s' in your model"
                           % layer.name)
                mace_check(layer.type in self._op_converters,
                           "Mace does not support caffe op type %s yet"
                           % layer.type)
                layer_names.add(layer.name)
                print("Begin convert ",layer.name)
                self._op_converters[layer.type](caffe_op)

    def add_tensor(self, name, shape, data_type, value, data_fromat = mace_pb2.DT_NCHW):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        tensor.data_format = data_fromat
        if data_type == mace_pb2.DT_INT32:
            tensor.int32_data.extend(value.flat)
        else:
            tensor.float_data.extend(value.flat)

    def convert_input(self, caffe_op):
        param = caffe_op.layer.input_param
        dims = list(param.shape[0].dim)
        if len(dims) == 4:
          mace_check(dims[0] == 1,
            "SGS only support N = 1 for 4 dims inputs, but input '%s' doesn't\n"
            % caffe_op.layer.name)
          if dims[1] == 1:
            #dims[1] = 3
            self.gray_model += 1
        self._input_name_shape_map[caffe_op.layer.top[0]]=dims

    def convert_general_op(self, caffe_op):
        op = self._mace_net_def.op.add()
        op.name = caffe_op.name
        op.type = caffe_op.type
        op.input.extend(caffe_op.layer.bottom)
        op.output.extend(caffe_op.layer.top)
        '''
        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.CAFFE.value
        '''
        ConverterUtil.add_data_format_arg(op, DataFormat.NCHW)
        return op

    def convert_conv2d(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.convolution_param
        is_depthwise = False
        if param.HasField(caffe_group_str) and param.group > 1:
            filter_data = caffe_op.blobs[0]
            if param.group == filter_data.shape[0] and filter_data.shape[1] == 1:
                is_depthwise = True
                caffe_op.blobs[0] = filter_data.reshape(1,
                                                        filter_data.shape[0],
                                                        filter_data.shape[2],
                                                        filter_data.shape[3])

        if is_depthwise:
            op.type = MaceOp.DepthwiseConv2d.name
        else:
            op.type = MaceOp.Conv2D.name
        self.add_stride_pad_kernel_arg(param, op)
        # dilation is specific for convolution in caffe
        dilations = [1, 1]
        if len(param.dilation) > 0:
            dilation_arg = op.arg.add()
            dilation_arg.name = MaceKeyword.mace_dilations_str
            if len(param.dilation) == 1:
                dilations = [param.dilation[0], param.dilation[0]]
            elif len(param.dilation) == 2:
                dilations = [param.dilation[0], param.dilation[1]]
            dilation_arg.ints.extend(dilations)

        filter_tensor_name = op.name + '_filter'
        filter_data = caffe_op.blobs[0]
        # gray pic case for I6E and M6
        if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
          input_name = op.input[0]
          if filter_data.shape[1] == 1 and is_depthwise == False and (input_name in self.input_node_array):
            if not ('RAWDATA_S16_NHWC' == self._input_name_format_map[input_name] or \
              'RAWDATA_F32_NHWC' == self._input_name_format_map[input_name]):
                   self._input_name_shape_map[input_name][1] = 3
                   [n,c,h,w] = filter_data.shape
                   dummy = np.zeros((n,2,h,w))
                   filter_data = np.concatenate((filter_data,dummy),axis=1)
                   self.gray_model -= 1
        self.add_tensor(filter_tensor_name, filter_data.shape,
                        mace_pb2.DT_FLOAT, filter_data)
        op.input.extend([filter_tensor_name])
        #we must have bias tensor
        if len(caffe_op.blobs) == 2:
            bias_data = caffe_op.blobs[1]
        else:
            ob = param.num_output
            bias_data = np.zeros(ob)
            # caffe of old version has 4-dimension bias, so reshape it
            # to single dimension
        bias_tensor_name = op.name + '_bias'
        self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                        mace_pb2.DT_FLOAT,
                        bias_data)
        op.input.extend([bias_tensor_name])

    def convert_deconv2d(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.convolution_param
        is_GroupConv = False

        if param.num_output != None:
            num_output = param.num_output
            num_output_arg = op.arg.add()
            num_output_arg.name = "num_output"
            num_output_arg.i = num_output

        if param.HasField(caffe_group_str) and param.group > 1:
            filter_data = caffe_op.blobs[0]
            print("group = ",param.group)
            print("filter_data.shape = ",filter_data.shape)
            if param.group == filter_data.shape[0] and filter_data.shape[1] == 1:
                is_depthwise = True
                caffe_op.blobs[0] = filter_data.reshape(1,
                                                        filter_data.shape[0],
                                                        filter_data.shape[2],
                                                        filter_data.shape[3])
                group_arg = op.arg.add()
                group_arg.name = MaceKeyword.mace_group_str
                group_arg.i = param.group
                op.type = MaceOp.DepthwiseDeconv2d.name
            else:
                is_GroupConv = True
                group_arg = op.arg.add()
                group_arg.name = MaceKeyword.mace_group_str
                group_arg.i = param.group
                op.type = MaceOp.Deconv2D.name
        else:
            op.type = MaceOp.Deconv2D.name

        self.add_stride_pad_kernel_arg(param, op)
        # dilation is specific for convolution in caffe
        dilations = [1, 1]
        if len(param.dilation) > 0:
            dilation_arg = op.arg.add()
            dilation_arg.name = MaceKeyword.mace_dilations_str
            if len(param.dilation) == 1:
                dilations = [param.dilation[0], param.dilation[0]]
            elif len(param.dilation) == 2:
                dilations = [param.dilation[0], param.dilation[1]]
            mace_check(dilations[0] == 1 and dilations[1] == 1,
                       "Mace only supports dilation == 1 deconvolution.")
            dilation_arg.ints.extend(dilations)

        filter_tensor_name = op.name + '_filter'
        if is_GroupConv:
            filter_data = caffe_op.blobs[0]
        else:
            filter_data = caffe_op.blobs[0].transpose(1,0,2,3)#reverse_dimensions == true
        self.add_tensor(filter_tensor_name, filter_data.shape,
                        mace_pb2.DT_FLOAT, filter_data)
        op.input.extend([filter_tensor_name])

        if len(caffe_op.blobs) == 2:
            bias_data = caffe_op.blobs[1]
        else:
            ob = param.num_output
            bias_data = np.zeros(ob)
            # caffe of old version has 4-dimension bias, so reshape it
            # to single dimension
        bias_tensor_name = op.name + '_bias'
        self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                        mace_pb2.DT_FLOAT,
                        bias_data)
        op.input.extend([bias_tensor_name])

    def convert_elementwise(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.eltwise_param

        op.type = MaceOp.Eltwise.name
        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = self.eltwise_type[param.operation].value
        if len(param.coeff) > 0:
            coeff_arg = op.arg.add()
            coeff_arg.name = 'coeff'
            coeff_arg.floats.extend(list(param.coeff))

    def convert_add(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.AddN.name

    def convert_activation(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.Activation.name

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b(self.activation_type[caffe_op.type].name)

        if caffe_op.type == 'PReLU':
            alpha_tensor_name = caffe_op.name + '_alpha'
            alpha_data = caffe_op.blobs[0]
            self.add_tensor(alpha_tensor_name, alpha_data.reshape(-1).shape,
                            mace_pb2.DT_FLOAT, alpha_data)
            op.input.extend([alpha_tensor_name])

        negative_slope = caffe_op.layer.relu_param.negative_slope
        if caffe_op.type == 'ReLU' and negative_slope != 0:
            param_arg = op.arg.add()
            param_arg.name = MaceKeyword.mace_activation_leakyrelu_coefficient_str  # noqa
            param_arg.f = caffe_op.layer.relu_param.negative_slope

            type_arg.s = six.b(ActivationType.LEAKYRELU.name)

        if caffe_op.type == 'Sigmoid':
            type_arg.s = six.b(ActivationType.SIGMOID.name)

    def convert_folded_batchnorm(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.BatchNorm.name
        scale_op = None
        for consumer in self._caffe_net.get_consumers(caffe_op.layer.top[0]):
            if consumer.type == 'Scale':
                scale_op = consumer

        # moving_average_fraction is used only when use_global_stats==False
        # As only support use_global_stats==True, param moving_average_fraction not needed
        param = caffe_op.layer.batch_norm_param
        if param.HasField('use_global_stats'):
            mace_check(param.use_global_stats == True,
                "batchnorm only support use_global_stats==True")
        if param.HasField('moving_average_fraction'):
            print("WARNING: Batchnorm no need to set moving_average_fraction as only support use_global_stats==True")

        epsilon_value = 1e-5
        if param.HasField('eps'):
            epsilon_value = param.eps

        #mace_check(scale_op is not None, "batchnorm is not followed by scale")
        if (scale_op is not None): # batchNorm + scale
          self._skip_ops.append(scale_op)
          if caffe_op.blobs[2][0] == 0:
            caffe_op.blobs[2][0] = 1
          mace_check(caffe_op.blobs[2][0] != 0, "batchnorm scalar is zero")
          mean_value = (1. / caffe_op.blobs[2][0]) * caffe_op.blobs[0]
          var_value = (1. / caffe_op.blobs[2][0]) * caffe_op.blobs[1]
          gamma_value = scale_op.blobs[0]
          beta_value = np.zeros_like(mean_value)
          if len(scale_op.blobs) == 2:
              beta_value = scale_op.blobs[1]

          scale_value = (
                  (1.0 / np.vectorize(math.sqrt)(var_value + epsilon_value)) *
                  gamma_value).reshape(-1)
          scale_value_inf = np.ones_like(scale_value) * np.inf
          if np.equal(scale_value_inf, scale_value).all():
            scale_value = np.ones_like(scale_value)
          offset_value = ((-mean_value * scale_value) + beta_value).reshape(-1)

          input_names = [op.name + '_scale', op.name + '_offset']
          self.add_tensor(input_names[0], scale_value.reshape(-1).shape,
                          mace_pb2.DT_FLOAT, scale_value)
          self.add_tensor(input_names[1], offset_value.reshape(-1).shape,
                          mace_pb2.DT_FLOAT, offset_value)
          op.input.extend([name for name in input_names])
          op.output[:] = scale_op.layer.top[:]
        else:
          mace_check(caffe_op.blobs[2][0] != 0, "batchnorm scalar is zero")
          mean_value = (1. / caffe_op.blobs[2][0]) * caffe_op.blobs[0]
          var_value = (1. / caffe_op.blobs[2][0]) * caffe_op.blobs[1]
          scale_value = (
                  (1.0 / np.vectorize(math.sqrt)(var_value + epsilon_value))).reshape(-1)
          scale_value_inf = np.ones_like(scale_value) * np.inf
          if np.equal(scale_value_inf, scale_value).all():
            scale_value = np.ones_like(scale_value)
          offset_value = (
                  (1.0 / np.vectorize(math.sqrt)(var_value + epsilon_value)) *
                  -mean_value).reshape(-1)
          input_names = [op.name + '_scale', op.name + '_offset']
          self.add_tensor(input_names[0], scale_value.reshape(-1).shape,
                          mace_pb2.DT_FLOAT, scale_value)
          self.add_tensor(input_names[1], offset_value.reshape(-1).shape,
                          mace_pb2.DT_FLOAT, offset_value)
          op.input.extend([name for name in input_names])
          #op.output[:] = scale_op.layer.top[:]

    def convert_pooling(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.pooling_param

        op.type = MaceOp.Pooling.name
        self.add_stride_pad_kernel_arg(param, op)
        pooling_type_arg = op.arg.add()
        pooling_type_arg.name = MaceKeyword.mace_pooling_type_str
        pooling_type_arg.i = self.pooling_type_mode[param.pool].value

    def convert_softmax(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.softmax_param
        op.type = MaceOp.Softmax.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if param.HasField('axis'):
            axis_arg.i = param.axis

    def convert_crop(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.crop_param
        op.type = MaceOp.Crop.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 2
        if param.HasField('axis'):
            axis_arg.i = param.axis
        offset_arg = op.arg.add()
        offset_arg.name = MaceKeyword.mace_offset_str
        if len(param.offset) > 0:
            offset_arg.ints.extend(list(param.offset))
        else:
            offset_arg.i = 0

    def convert_concat(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.concat_param
        op.type = MaceOp.Concat.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if param.HasField('axis'):
            axis_arg.i = param.axis
        elif param.HasField('concat_dim'):
            axis_arg.i = param.concat_dim

    def convert_slice(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.Slice.name

        if caffe_op.layer.HasField('slice_param'):
            param = caffe_op.layer.slice_param
            '''
            mace_check(not param.HasField('axis') or param.axis == 1
                       or param.axis == -3,
                       "Mace do not support slice with axis %d" % param.axis)
            mace_check(len(param.slice_point) == 0,
                       "Mace do not support slice with slice_point")
            '''
            if param.HasField('slice_dim'):
                axis_arg = op.arg.add()
                axis_arg.name = MaceKeyword.mace_axis_str
                axis_arg.i = param.slice_dim
            if param.HasField('axis'):
                axis_arg = op.arg.add()
                axis_arg.name = MaceKeyword.mace_axis_str
                axis_arg.i = param.axis
            if len(param.slice_point) != 0:
                 slice_point_arg = op.arg.add()
                 slice_point_arg.name = 'slice_point'
                 slice_point_arg.ints.extend(param.slice_point)

    def convert_split(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.Split.name
        numSplits_arg = op.arg.add()
        numSplits_arg.name = MaceKeyword.mace_num_split_str
        numSplits_arg.i = len(op.output_shape)

    def convert_interp(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.interp_param
        mace_check(param.HasField("height") and param.HasField("width"),
                   'Only support bilinear interp with height and width')
        op.type = MaceOp.ResizeBilinear.name

        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_resize_size_str
        size_value = np.array([param.height, param.width], dtype=np.int32)
        size_arg.ints.extend(size_value)

    def convert_fully_connected(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.inner_product_param
        op.type = MaceOp.FullyConnected.name
        '''
        mace_check((param.axis == 1 or param.axis == -3)
                   and not param.transpose,
                   "Do not support non-default axis and transpose")
        mace_check(caffe_op.blobs[0].ndim in [2, 4],
                   "Unexpected fc weigth ndim.")
        if caffe_op.blobs[0].ndim == 4:
            mace_check(list(caffe_op.blobs[0].shape[:2]) == [1, 1],
                       "Do not support 4D weight with shape [1, 1, *, *]")
        '''
        #add by sigmastar
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if param.HasField('axis'):
            axis_arg.i = param.axis

        transpose = False
        if param.HasField('transpose'):
            transpose = param.transpose

        weight_tensor_name = op.name + '_weight'
        if transpose == False:
            weight_data = caffe_op.blobs[0]
        else:
            weight_data = caffe_op.blobs[0].transpose(1,0)

        #change FC weight shape to 4D dim
        weight_data.shape = [weight_data.shape[0],weight_data.shape[1],1,1]
        self.add_tensor(weight_tensor_name, weight_data.shape,
                        mace_pb2.DT_FLOAT,
                        weight_data)
        op.input.extend([weight_tensor_name])
        if len(caffe_op.blobs) == 2:
            bias_data = caffe_op.blobs[1]
        else:
            ob = weight_data.shape[0]
            bias_data = np.zeros(ob)
            # caffe of old version has 4-dimension bias, so reshape it
            # to single dimension
        bias_tensor_name = op.name + '_bias'
        self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                        mace_pb2.DT_FLOAT,
                        bias_data)
        op.input.extend([bias_tensor_name])

    def convert_scale(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = 'Scale'
        param = caffe_op.layer.scale_param
        scale_op_name = op.name
        op.name = scale_op_name + '_prod'
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if param.HasField('axis'):
            axis_arg.i = param.axis
        axis_arg.i = 4 + axis_arg.i if axis_arg.i < 0 else axis_arg.i

        if len(caffe_op.blobs) != 0:
            scale_tensor_name = scale_op_name + '_scale'
            scale_data = caffe_op.blobs[0]
            if len(scale_data.shape) != 4:
                #expend dim to 4 ,bcaus mul input2 must be as same as input1
                num = 4
                tmp_shape = [1,1,1,1]
                for i in six.moves.range(len(scale_data.shape)-1,-1,-1):
                    tmp_shape[num-1] = scale_data.shape[i]
                    num = num - 1
                shape = [tmp_shape[0],tmp_shape[3],tmp_shape[1],tmp_shape[2]]
                tmp_shape = tmp_shape if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6' else shape
                self.add_tensor(scale_tensor_name, tmp_shape,
                                mace_pb2.DT_FLOAT, scale_data,mace_pb2.DT_NHWC)
                op.input.extend([scale_tensor_name])
            else:
                self.add_tensor(scale_tensor_name, scale_data.shape,
                                mace_pb2.DT_FLOAT, scale_data,mace_pb2.DT_NCHW)
                op.input.extend([scale_tensor_name])

            if len(caffe_op.blobs) == 2:
                bias_tensor_name = scale_op_name + '_offset'
                bias_data = caffe_op.blobs[1]
                # caffe of old version has 4-dimension bias, so reshape it
                # to single dimension ???
                #expend dim to 4 bcaus add input2 must be as same as input1
                if len(bias_data.shape) != 4:
                    num = 4
                    tmp_shape = [1,1,1,1]
                    for i in six.moves.range(len(bias_data.shape)-1,-1,-1):
                        tmp_shape[num-1] = bias_data.shape[i]
                        num = num - 1
                    shape = [tmp_shape[0],tmp_shape[3],tmp_shape[1],tmp_shape[2]]
                    tmp_shape = tmp_shape if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6' else shape
                    self.add_tensor(bias_tensor_name, tmp_shape,
                                    mace_pb2.DT_FLOAT,
                                    bias_data,mace_pb2.DT_NHWC)
                    op.input.extend([bias_tensor_name])
                    op.type = MaceOp.BatchNorm.name
                else:
                    self.add_tensor(bias_tensor_name, bias_data.shape,
                                    mace_pb2.DT_FLOAT,
                                    bias_data,mace_pb2.DT_NCHW)
                    op.input.extend([bias_tensor_name])
                    op.type = MaceOp.BatchNorm.name

                # biasadd_op = self._mace_net_def.op.add()
                # biasadd_op.name = scale_op_name + '_biasadd'
                # biasadd_op.type = MaceOp.BiasAdd.name
                # biasadd_op.output.extend(op.output)
                # op.output[:] = [op.output[0] + '_prod_output']
                # biasadd_op.input.extend(op.output)
                # biasadd_op.input.extend([op.input[2]])
                # biasadd_op.output_shape.extend(op.output_shape)
                # del op.input[2]

                # data_type_arg = biasadd_op.arg.add()
                # data_type_arg.name = 'T'
                # data_type_arg.i = self._option.data_type

                # ConverterUtil.add_data_format_arg(biasadd_op,
                #                                   DataFormat.NCHW)

    def convert_channel_shuffle(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.shuffle_channel_param
        op.type = MaceOp.ChannelShuffle.name

        group_arg = op.arg.add()
        group_arg.name = MaceKeyword.mace_group_str
        group_arg.i = 1
        if param.HasField('group'):
            group_arg.i = param.group

    def convert_permute(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.permute_param
        op.type = MaceOp.Transpose.name


        permute_shape = op.name + '_shape'
        permute_value = param.order
        self.add_tensor(permute_shape, np.shape(permute_value),
                        mace_pb2.DT_INT32, np.array(permute_value))
        op.input.extend([permute_shape])

        dims_arg = op.arg.add()
        dims_arg.name = MaceKeyword.mace_dims_str
        dims_arg.ints.extend(list(param.order))


    def convert_flatten(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.flatten_param
        op.type = MaceOp.Reshape.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if param.HasField('axis'):
            axis_arg.i = param.axis
        axis_arg.i = 4 + axis_arg.i if axis_arg.i < 0 else axis_arg.i

        end_axis_arg = op.arg.add()
        end_axis_arg.name = MaceKeyword.mace_end_axis_str
        end_axis_arg.i = -1
        if param.HasField('end_axis'):
            end_axis_arg.i = param.end_axis

    def convert_prior_box(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.prior_box_param
        op.type = MaceOp.PriorBox.name

        min_size_arg = op.arg.add()
        min_size_arg.name = MaceKeyword.mace_min_size_str
        min_size_arg.floats.extend(list(param.min_size))
        max_size_arg = op.arg.add()
        max_size_arg.name = MaceKeyword.mace_max_size_str
        max_size_arg.floats.extend(list(param.max_size))
        flip_arg = op.arg.add()
        flip_arg.name = MaceKeyword.mace_flip_str
        flip_arg.i = 1
        if param.HasField('flip'):
            flip_arg.i = int(param.flip)
        aspect_ratio = [1.0]
        for i in param.aspect_ratio:
            already_exist = False
            for ar in aspect_ratio:
                if abs(i - ar) < 1e-6:
                    already_exist = True
                    break
            if not already_exist:
                aspect_ratio.append(i)
                if flip_arg.i:
                    aspect_ratio.append(1.0 / i)
        aspect_ratio_arg = op.arg.add()
        aspect_ratio_arg.name = MaceKeyword.mace_aspect_ratio_str
        aspect_ratio_arg.floats.extend(list(aspect_ratio))
        clip_arg = op.arg.add()
        clip_arg.name = MaceKeyword.mace_clip_str
        clip_arg.i = 0
        if param.HasField('clip'):
            clip_arg.i = int(param.clip)
        variance_arg = op.arg.add()
        variance_arg.name = MaceKeyword.mace_variance_str
        variance_arg.floats.extend(list(param.variance))
        offset_arg = op.arg.add()
        offset_arg.name = MaceKeyword.mace_offset_str
        offset_arg.f = 0.5
        if param.HasField('offset'):
            offset_arg.f = param.offset
        step_h_arg = op.arg.add()
        step_h_arg.name = MaceKeyword.mace_step_h_str
        step_h_arg.f = 0
        if param.HasField('step_h'):
            mace_check(not param.HasField('step'),
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            step_h_arg.f = param.step_h
            mace_check(step_h_arg.f > 0, "step_h should be larger than 0.")
        step_w_arg = op.arg.add()
        step_w_arg.name = MaceKeyword.mace_step_w_str
        step_w_arg.f = 0
        if param.HasField('step_w'):
            mace_check(not param.HasField('step'),
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            step_w_arg.f = param.step_w
            mace_check(step_w_arg.f > 0, "step_w should be larger than 0.")

        if param.HasField('step'):
            mace_check(not param.HasField('step_h') and not param.HasField('step_w'),  # noqa
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            mace_check(param.step > 0, "step should be larger than 0.")
            step_h_arg.f = param.step
            step_w_arg.f = param.step

    def convert_prior_box_RFC(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.prior_box_param
        #op.type = MaceOp.PriorBox.name
        op.type = caffe_op.type
        min_size_arg = op.arg.add()
        min_size_arg.name = MaceKeyword.mace_min_size_str
        min_size_arg.floats.extend(list(param.min_size))
        max_size_arg = op.arg.add()
        max_size_arg.name = MaceKeyword.mace_max_size_str
        max_size_arg.floats.extend(list(param.max_size))
        flip_arg = op.arg.add()
        flip_arg.name = MaceKeyword.mace_flip_str
        flip_arg.i = 1
        if param.HasField('flip'):
            flip_arg.i = int(param.flip)
        aspect_ratio = []
        for i in param.aspect_ratio:
            already_exist = False
            for ar in aspect_ratio:
                if abs(i - ar) < 1e-6:
                    already_exist = True
                    break
            if not already_exist:
                aspect_ratio.append(i)
                if flip_arg.i:
                    aspect_ratio.append(1.0 / i)
        aspect_ratio_arg = op.arg.add()
        aspect_ratio_arg.name = MaceKeyword.mace_aspect_ratio_str
        aspect_ratio_arg.floats.extend(list(aspect_ratio))
        clip_arg = op.arg.add()
        clip_arg.name = MaceKeyword.mace_clip_str
        clip_arg.i = 0
        if param.HasField('clip'):
            clip_arg.i = int(param.clip)
        variance_arg = op.arg.add()
        variance_arg.name = MaceKeyword.mace_variance_str
        variance_arg.floats.extend(list(param.variance))
        offset_arg = op.arg.add()
        offset_arg.name = MaceKeyword.mace_offset_str
        offset_arg.f = 0.5
        if param.HasField('offset'):
            offset_arg.f = param.offset
        step_h_arg = op.arg.add()
        step_h_arg.name = MaceKeyword.mace_step_h_str
        step_h_arg.f = 0
        if param.HasField('step_h'):
            mace_check(not param.HasField('step'),
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            step_h_arg.f = param.step_h
            mace_check(step_h_arg.f > 0, "step_h should be larger than 0.")
        step_w_arg = op.arg.add()
        step_w_arg.name = MaceKeyword.mace_step_w_str
        step_w_arg.f = 0
        if param.HasField('step_w'):
            mace_check(not param.HasField('step'),
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            step_w_arg.f = param.step_w
            mace_check(step_w_arg.f > 0, "step_w should be larger than 0.")

        if param.HasField('step'):
            mace_check(not param.HasField('step_h') and not param.HasField('step_w'),  # noqa
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            mace_check(param.step > 0, "step should be larger than 0.")
            step_h_arg.f = param.step
            step_w_arg.f = param.step
    def convert_reshape(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.reshape_param
        op.type = MaceOp.Reshape.name

        dim_arg = op.arg.add()
        dim_arg.name = MaceKeyword.mace_dim_str
        dim_arg.ints.extend(list(param.shape.dim))

        axis_arg = op.arg.add()
        axis_arg.name = 'reshape_' + MaceKeyword.mace_axis_str
        axis_arg.i = 0
        if param.HasField('axis'):
            axis_arg.i = param.axis

        num_axes_arg = op.arg.add()
        num_axes_arg.name = MaceKeyword.mace_num_axes_str
        num_axes_arg.i = -1
        if param.HasField('num_axes'):
            num_axes_arg.i = param.num_axes

    '''add by sigmastar'''
    def convert_ArgMax(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = 'ArgMax'
        param = caffe_op.layer.argmax_param
        out_max_val_arg = op.arg.add()
        out_max_val_arg.name = "out_max_val"
        out_max_val_arg.i = 0 #default is 0
        top_k_arg = op.arg.add()
        top_k_arg.name = "top_k"
        top_k_arg.i = 1 #default is 1
        axis_arg = op.arg.add()
        axis_arg.name = "axis"
        axis_arg.i = 0 #default is 1

        if param.HasField('out_max_val'):
            out_max_val_arg.i = int(param.out_max_val)
            mace_check(out_max_val_arg.i == 0, "only output index")
        if param.HasField('top_k'):
            top_k_arg.i = int(param.top_k)
            mace_check(top_k_arg.i == 1, "only support top 1")
        if param.HasField('axis'):
            axis_arg.i = int(param.axis)

    def convert_Axpy(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = 'Axpy'

    def convert_CReLU(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = 'CReLU'
        param = caffe_op.layer.crelu_param
        neg_slope_arg = op.arg.add()
        neg_slope_arg.name = "negative_slope"
        neg_slope_arg.f = 0 #default is 0
        concat_axis_arg = op.arg.add()
        concat_axis_arg.name = "concat_axis"
        concat_axis_arg.i = 1 #default is 1
        if param.HasField('negative_slope'):
            neg_slope_arg.f = param.negative_slope
        if param.HasField('concat_axis'):
            concat_axis_arg.i = int(param.concat_axis)


    def convert_ConvolutionDepthwise(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.convolution_param
        if caffe_op.blobs != []:
          filter_data = caffe_op.blobs[0]
          caffe_op.blobs[0] = filter_data.reshape(1,
                                                  filter_data.shape[0],
                                                  filter_data.shape[2],
                                                  filter_data.shape[3])
          filter_tensor_name = op.name + '_filter'
          filter_data = caffe_op.blobs[0]
          self.add_tensor(filter_tensor_name, filter_data.shape,
                          mace_pb2.DT_FLOAT, filter_data)
          op.input.extend([filter_tensor_name])
        op.type = MaceOp.DepthwiseConv2d.name
        self.add_stride_pad_kernel_arg(param, op)
        # dilation is specific for convolution in caffe
        dilations = [1, 1]
        if len(param.dilation) > 0:
            dilation_arg = op.arg.add()
            dilation_arg.name = MaceKeyword.mace_dilations_str
            if len(param.dilation) == 1:
                dilations = [param.dilation[0], param.dilation[0]]
            elif len(param.dilation) == 2:
                dilations = [param.dilation[0], param.dilation[1]]
            dilation_arg.ints.extend(dilations)


        #we must have bias tensor
        if len(caffe_op.blobs) == 2:
            bias_data = caffe_op.blobs[1]
        else:
            ob = param.num_output
            bias_data = np.zeros(ob)
            # caffe of old version has 4-dimension bias, so reshape it
            # to single dimension
        bias_tensor_name = op.name + '_bias'
        self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                        mace_pb2.DT_FLOAT,
                        bias_data)
        op.input.extend([bias_tensor_name])

    def convert_Clip(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.clip_param
        is_relux = False # (min_value=0,max_value=6)
        is_clip = False  # (min_value,max_value)
        is_maximum = False # (min_value,-)
        is_minimum = False # (-,max_value)
        default_min = np.finfo(np.float32).min
        default_max = np.finfo(np.float32).max
        #op.type = 'Clip'
        if param.HasField('min'):
           min = param.min
           min_value = float(min)
        if param.HasField('max'):
           max = param.max
           max_value = float(max)

        # Classification
        if min_value == default_min and max_value != default_max:
            is_minimum = True # (-,max_value)
        elif min_value != default_min and max_value == default_max:
            is_maximum = True # (min_value,-)
        elif min_value != 0 and max_value != 6:
            is_clip = True  # (min_value,max_value)
        else:
            is_relux = True # (min_value=0,max_value=6)

        # elt type
        if is_minimum:
            op.type = MaceOp.Eltwise.name
            type_arg = op.arg.add()
            type_arg.name = MaceKeyword.mace_element_type_str
            type_arg.i = 4
            if node.op_type == OnnxOpType.Clip.name:
                coeff_arg = op.arg.add()
                coeff_arg.name = MaceKeyword.mace_coeff_str
                coeff_arg.floats.extend([min_value, max_value])
        elif is_maximum:
            op.type = MaceOp.Eltwise.name
            type_arg = op.arg.add()
            type_arg.name = MaceKeyword.mace_element_type_str
            type_arg.i = 5
            if node.op_type == OnnxOpType.Clip.name:
                coeff_arg = op.arg.add()
                coeff_arg.name = MaceKeyword.mace_coeff_str
                coeff_arg.floats.extend([min_value, max_value])
        # clip type
        elif is_clip or is_relux:
            op.type = 'Clip'
            coeff_arg = op.arg.add()
            coeff_arg.name = MaceKeyword.mace_coeff_str
            coeff_arg.floats.extend([min_value, max_value])

    def convert_Dropout(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.dropout_param
        op.type = 'Dropout'
        dropout_ratio = op.arg.add()
        dropout_ratio.name = 'dropout_ratio'
        dropout_ratio.f = 0.5 #default value
        if param.HasField('dropout_ratio'):
           dropout_ratio.f = param.dropout_ratio
        if param.HasField('scale_train'):
           scale_train = op.arg.add()
           scale_train.name = 'scale_train'
           scale_train.i = int(param.scale_train)
    def convert_Normalize(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = "Normalize"
        param = caffe_op.layer.norm_param

        across_spatial_arg = op.arg.add()
        across_spatial_arg.name = "across_spatial"
        across_spatial_arg.i = 1 #default is true
        if param.HasField('across_spatial'):
            across_spatial_arg.i = int(param.across_spatial)

        channel_shared_arg = op.arg.add()
        channel_shared_arg.name = "channel_shared"
        channel_shared_arg.i = 1 #default is true
        if param.HasField('channel_shared'):
            channel_shared_arg.i = int(param.channel_shared)
        scale_data = caffe_op.blobs[0]
        if scale_data.size == 1 and scale_data.shape == ():
            scale_data.shape = 1
        scale_tensor_name = op.name + '_scale'
        self.add_tensor(scale_tensor_name, scale_data.shape,
                        mace_pb2.DT_FLOAT, scale_data)
        op.input.extend([scale_tensor_name])
        if param.HasField('eps'):
            eps = param.eps
        else:
            eps = 1e-10
        eps_arg = op.arg.add()
        eps_arg.name = 'eps'
        eps_arg.f = eps
    def convert_Upsample(self, caffe_op):
     op = self.convert_general_op(caffe_op)
     op.type = "Upsample"
     param = caffe_op.layer.upsample_param
     scale_arg = op.arg.add()
     scale_arg.name = "scale"
     scale_arg.i = 1 #default is 1
     if param.HasField('scale'):
         scale_arg.i = int(param.scale)
    def convert_Reorg(self, caffe_op):
     op = self.convert_general_op(caffe_op)
     op.type = "Reorg"
     param = caffe_op.layer.reorg_param
     stride_arg = op.arg.add()
     stride_arg.name = "stride"
     stride_arg.i = 1 #default is true
     if param.HasField('stride'):
         stride_arg.i = int(param.stride)
     mace_check( stride_arg.i == 2, "only support stride is 2")

    def convert_Reverse(self, caffe_op):
     op = self.convert_general_op(caffe_op)
     op.type = "Reverse"
     param = caffe_op.layer.reverse_param
     axis_arg = op.arg.add()
     axis_arg.name = MaceKeyword.mace_axis_str
     axis_arg.i = 0
     if param.HasField('axis'):
         axis_arg.i = param.axis


    def convert_LSTM(self, caffe_op):
     folder = './lstm_data'
     ConverterUtil.mkdir(folder)
     op = self.convert_general_op(caffe_op)
     param = caffe_op.layer.recurrent_param
     op.type = 'LSTM'
     num_output_arg = op.arg.add()
     num_output_arg.name = 'num_output'
     num_output_arg.i = param.num_output
     weight_data1 = caffe_op.blobs[0]
     weight_data2 = caffe_op.blobs[2]
     weight_data = np.concatenate((weight_data1, weight_data2), axis=1)
     weight_data.shape = [weight_data.shape[0],weight_data.shape[1],1,1]
     bias_data = caffe_op.blobs[1]
     file_name = folder + '/weight_biase_data#' + str(self._lstm_num)
     self._lstm_num += 1
     np.savez(file_name, weight = weight_data,bias = bias_data)

    def convert_Threshold(self, caffe_op):
     op = self.convert_general_op(caffe_op)
     op.type = "Threshold"
     param = caffe_op.layer.threshold_param
     threshold_data_name = "threshold_data"
     data = 0 #default is true
     if param.HasField('threshold'):
         data = param.threshold
     self.add_tensor(threshold_data_name, [1], mace_pb2.DT_FLOAT, np.array(data))
     op.input.extend([threshold_data_name])

    def convert_Tile(self, caffe_op):
     op = self.convert_general_op(caffe_op)
     op.type = "Tile"
     param = caffe_op.layer.tile_param

     tiles_arg = op.arg.add()
     tiles_arg.name = "tiles"
     tiles_arg.i = param.tiles

     axis_arg = op.arg.add()
     axis_arg.name = MaceKeyword.mace_axis_str
     axis_arg.i = 1
     if param.HasField('axis'):
         axis_arg.i = param.axis

    def convert_ContinuationIndicator(self, caffe_op):
     op = self.convert_general_op(caffe_op)
     op.type = "ContinuationIndicator"
     param = caffe_op.layer.continuation_indicator_param

     time_step_arg = op.arg.add()
     time_step_arg.name = "time_step"
     time_step_arg.i = 0
     if param.HasField('time_step'):
         time_step_arg.i = param.time_step

     batch_size_arg = op.arg.add()
     batch_size_arg.name = "batch_size"
     batch_size_arg.i = 0
     if param.HasField('batch_size'):
         batch_size_arg.i = param.batch_size
             # only support 1 batch
     mace_check( batch_size_arg.i == 1, "only support 1 batch")

    def convert_ROIPooling(self, caffe_op):
     op = self.convert_general_op(caffe_op)
     op.type = "ROIPooling"
     param = caffe_op.layer.roi_pooling_param

     pooled_h_arg = op.arg.add()
     pooled_h_arg.name = 'pooled_h'
     pooled_h_arg.i = 0
     if param.HasField('pooled_h'):
       pooled_h_arg.i = param.pooled_h

     pooled_w_arg = op.arg.add()
     pooled_w_arg.name = 'pooled_w'
     pooled_w_arg.i = 0
     if param.HasField('pooled_w'):
       pooled_w_arg.i = param.pooled_w

     spatial_scale_arg = op.arg.add()
     spatial_scale_arg.name = 'spatial_scale'
     spatial_scale_arg.f = 1
     if param.HasField('spatial_scale'):
       spatial_scale_arg.f = param.spatial_scale

    def convert_Power(self, caffe_op):
     op = self.convert_general_op(caffe_op)
     op.type = "Power"
     param = caffe_op.layer.power_param

     power_arg = op.arg.add()
     power_arg.name = 'power'
     power_arg.f = 1.0
     if param.HasField('power'):
       power_arg.f = param.power

     scale_arg = op.arg.add()
     scale_arg.name = 'scale'
     scale_arg.f = 1.0
     if param.HasField('scale'):
       scale_arg.f = param.scale

     shift_arg = op.arg.add()
     shift_arg.name = 'shift'
     shift_arg.f = 0.0
     if param.HasField('shift'):
       shift_arg.f = param.shift
     scale_value = np.array([scale_arg.f])
     offset_value = np.array([shift_arg.f])
     input_names = [op.name + '_scale', op.name + '_offset']
     self.add_tensor(input_names[0], [1],
                   mace_pb2.DT_FLOAT, scale_value)
     self.add_tensor(input_names[1], [1],
                   mace_pb2.DT_FLOAT, offset_value)
     op.input.extend([name for name in input_names])

    def conver_SGS_SSD_Postprocess(self, caffe_op):
      op = self.convert_general_op(caffe_op)
      pass
    def conver_SGS_YoloV2_Postprocess(self, caffe_op):
      op = self.convert_general_op(caffe_op)
      pass
    def conver_SGS_YoloV3_Postprocess(self, caffe_op):
      op = self.convert_general_op(caffe_op)
      pass
    def conver_SGS_LanceNet_Postprocess(self, caffe_op):
      op = self.convert_general_op(caffe_op)
      pass
    def conver_SGS_FDA_Postprocess(self, caffe_op):
      op = self.convert_general_op(caffe_op)
      pass
    def conver_SGS_CAFFE_SSD_Postprocess(self, caffe_op):
      op = self.convert_general_op(caffe_op)
      pass
    def convert_Custom(self, caffe_op):
     pass
