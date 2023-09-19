import os
import sys
import numpy as np

import pdb
import six
import struct
import collections
from ctypes import *
import ctypes
import shutil
from mace.python.tools import utility
from mace.python.tools.convert_util import mace_check
from mace.python.tools.convert_util import getIPUVersion

from mace.python.tools.converter_tool.transformer import Transformer
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.sgs_tflite import SGSModel_transform_tflite
from mace.python.tools.sgs_onnx import SGSModel_transform_onnx
from mace.python.tools.sgs_onnx import SGSModel_transform_onnx_S
from mace.python.tools.sgs_caffe import SGSModel_transform
from mace.python.tools.sgs_caffe import SGSModel_transform_caffe_S

from mace.proto import mace_pb2
from third_party.python import flatbuffers
from third_party import tflite
import copy


class CheckSGSModel(object):
    def __init__(self, net, TensorMap, ValueMap):
        self._net = net
        self._tensor_name_shape_map = TensorMap
        self._tensor_name_value_map = ValueMap
        self._check_op = {
            'ADD':self.check_eltwise,
            'SUB':self.check_eltwise,
            'MUL':self.check_eltwise,
            'DIV':self.check_eltwise,
            'RESHAPE':self.check_reshape,
            'TRANSPOSE':self.check_transpose,
            'SUM':self.check_reduce,
            'MEAN':self.check_reduce,
            #'SPLIT':self.check_split,
            'SPLIT_V':self.check_split,
            'CONCATENATION':self.check_concat,
            'TILE':self.check_tile,
         }
        self._check_type = ['ADD','SUB','MUL','DIV']

    def run(self):
        for op in self._net.op:
            type = op.type
            if type in self._check_op.keys():
                self._check_op[type](op)

    def get_shape_by_name(self, name):
        try:
          shape = self._tensor_name_shape_map[name]
        except Exception:
          six.print_("can't find name in map",name)
          assert 0
        return shape

    def get_value_by_name(self, name):
        try:
          value = self._tensor_name_value_map[name]
        except Exception:
          six.print_("can't find name in map",name)
          assert 0
        return value

    def check_eltwise(self, op):
        #elt outshape should be the same as left inshape
        output_shape = self.get_shape_by_name(op.output[0])
        expected_shape1 = self.get_shape_by_name(op.input[0])
        if output_shape != expected_shape1:
          if len(op.input) > 1:
            expected_shape2 = self.get_shape_by_name(op.input[1])
            if output_shape != expected_shape2:
                if output_shape == list(np.multiply(expected_shape1, expected_shape2)):
                    return
                else:
                    print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.output[0]) +":" + str(output_shape))
          else:
            print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.output[0]) +":" + str(output_shape))

    def check_reshape(self, op):
        #reshape outshape should be the same as second inshape
        output_shape = self.get_shape_by_name(op.output[0])
        expected_shape = self.get_value_by_name(op.input[1])
        if output_shape != expected_shape:
            if expected_shape[-1] == -1:
                [i0,i1,i2,i3] = self.get_shape_by_name(op.input[0])
                if output_shape[-1] != i1*i2*i3:
                    print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.type) + '\n'  + str(op.output[0]) +":" + str(output_shape))
            else:
                print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.type) + '\n'  + str(op.output[0]) +":" + str(output_shape))

    def check_transpose(self, op):
        #transpose outshape should equal to the result after input0 is reversed in the order of input1
        output_shape = self.get_shape_by_name(op.output[0])
        if len(output_shape) == 4:
            in_shape = self.get_shape_by_name(op.input[0])
            order = self.get_value_by_name(op.input[1])
            expected_shape = []
            for i in six.moves.range(len(order)):
                expected_shape.append(in_shape[order[i]])
            if output_shape != expected_shape:
                print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.output[0]) +":" + str(output_shape))

    def check_reduce(self, op):
        #reduce outshape dimension that can read the axis should be 1
        output_shape = self.get_shape_by_name(op.output[0])
        in_shape = self.get_shape_by_name(op.input[0])
        arg = op.arg
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == 'keepdims': # "keepdims"
                keepdim = arg[i].i
        axis = self.get_value_by_name(op.input[1])
        if keepdim:
            for i in six.moves.range(len(in_shape)):
                expected_shape = [1 if i in axis else in_shape[i]]
        else:
            expected_shape = []
            for i in six.moves.range(len(in_shape)):
                if i not in axis:
                    expected_shape.append(in_shape[i])
        if output_shape != expected_shape:
            print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.output[0]) +":" + str(output_shape))


    def check_split(self, op):
        #The sum of the output axis dimensions is equal to the input axis dimensions
        arg = op.arg
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == 'axis': # "keepdims"
                axis = arg[i].i
        SumResult = 0
        for j in six.moves.range(len(op.output)):
            output_shape = self.get_shape_by_name(op.output[j])
            SumResult += output_shape[axis]
        in_shape = self.get_shape_by_name(op.input[0])
        if in_shape[axis] != SumResult:
            print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.output[0]))


    def check_concat(self, op):
        #The sum of the output axis dimensions is equal to the input axis dimensions
        arg = op.arg
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == 'axis': # "keepdims"
                axis = arg[i].i
        SumResult = 0
        for j in six.moves.range(len(op.input)):
            input_shape = self.get_shape_by_name(op.input[j])
            SumResult += input_shape[axis]
        output_shape = self.get_shape_by_name(op.output[0])
        if output_shape[axis] != SumResult:
            print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.output[0]))


    def check_tile(self, op):
        #tile outshape should equal to the result after input0 * input1
        input_shape = self.get_shape_by_name(op.input[0])
        tile_param = self.get_value_by_name(op.input[1])
        output_shape = self.get_shape_by_name(op.output[0])
        expected_shape = list(np.multiply(np.array(input_shape),np.array(tile_param)))
        if output_shape != expected_shape:
            print("[WARNING!]shape transposed does not meet expectations,pls check it! tensor informations:\n" + str(op.type) + '\n' + str(op.output[0]) +":" + str(output_shape))

