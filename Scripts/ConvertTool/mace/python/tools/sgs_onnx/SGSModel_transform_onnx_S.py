import numpy as np
import six
import math
import pdb
import copy
import configparser
import shutil
import string
import os
from mace.proto import mace_pb2
from mace.python.tools.convert_util import mace_check
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools import utility
from mace.python.tools.convert_util import *
from mace.python.tools.converter_tool.transformer import Transformer
from third_party import tflite
from collections import Counter

INPUT_CONFIG_INI = None

class TransformToSchema(object):
    """A class for transform mace model to schema model.
    """
    #ONNX only use paddingType:CAFFE, ONNXINSIDE, ONNXOUTSIDE to create sgs_float model,
    #and the "DICT NUMBER" is based on sgs_schema.fbs
    #                        0     1      2        3            4
    #enum Padding : byte { SAME, VALID, CAFFE, ONNXINSIDE, ONNXOUTSIDE, }
    #The number is using for create operator:
    #SGSModel_converter_from_Mace.py -> builtin_AVERAGE_POOL_2D(), builtin_MAX_POOL_2D()
    pooling_paddingType = {
        'CAFFE': 2,
        'ONNXINSIDE': 3,
        'ONNXOUTSIDE': 4,
    }

    def __init__(self,  model,inputName,inputNameShapeMap,outputName,input_name_format_map,input_pack_model,output_pack_model):
        self._splitOP_transform = {
            'Activation':self.split_Activation,
            'ArgMax':self.split_ArgMax,
            'And':self.split_And,
            'Atan':self.split_Atan,
            'BatchNorm':self.split_BatchNorm,
            'Cast':self.split_Cast,
            'Concat':self.split_Concat,
            'Conv1D': self.split_Conv1D,
            'Conv2D':self.split_Conv2D,
            'Conv3D':self.split_Conv3D,
            'Crop':self.split_Crop,
            'CReLU':self.split_CReLU,
            'Clip':self.split_Clip,
            'Cos':self.split_Cos,
            'ConvTranspose3D':self.split_ConvTranspose3D,
            'ConvTranspose1D':self.split_ConvTranspose1D,
            'Deconv2D':self.split_Deconv2D,
            'DepthwiseConv1d': self.split_DepthwiseConv1d,
            'DepthwiseConv2d':self.split_DepthwiseConv2d,
            'DepthwiseDeconv2d':self.split_Deconv2D,
            # 'Dropout':self.split_Dropout,
            'DepthToSpace':self.split_DepthToSpace,
            'Expand':self.split_Expand,
            'Eltwise':self.split_Eltwise,
            'Exp':self.split_Exp,
            'Erf':self.split_Erf,
            'Einsum':self.split_Einsum,
            'FullyConnected':self.split_FullyConnected,
            'Gather':self.split_Gather,
            'GreaterOrEqual': self.split_GreaterOrEqual,
            'HardSigmoid': self.split_HardSigmoid,
            'InstanceNorm': self.split_InstanceNorm,
            'LSTM':self.split_LSTM,
            'Log':self.split_Log,
            'Less':self.split_Less,
            'GRU':self.split_GRU,
            'LogSoftmax':self.split_LogSoftmax,
            'MatMul':self.split_BatchMatMul,
            'Pad':self.split_Pad,
            'Pooling':self.split_Pooling,
            'PriorBox':self.split_PriorBox,
            'Reshape':self.split_Reshape,
            'Where':self.split_Where,
            'Greater': self.split_Greater,
            'Elu': self.split_Elu,
            'ScatterND': self.split_scatternd,
            'Slice': self.split_Slice,
            'Split': self.split_Split,
            'Softmax':self.split_Softmax,
            'Softplus':self.split_Softplus,
            'Transpose':self.split_Transpose,
            'Tile':self.split_Tile,
            #'Normalize':self.split_Normalize,
            'Not':self.split_Not,
            #'Reorg':self.split_Reorg,
            'Upsample':self.split_Upsample,
            'Threshold':self.split_Threshold,
            'Reduce':self.split_Reduce,
            'Reduce_mean':self.split_Reduce_Mean,
            'ResizeBilinear':self.split_ResizeBilinear,
            'ResizeNearestNeighbor':self.split_ResizeNearestNeighbor,
            'ReduceL2':self.split_ReduceL2,
            'SpaceToDepth':self.split_SpaceToDepth,
            'Sin':self.split_Sin,
            'SGS_SSD_Postprocess':self.split_SGS_CAFFE_SSD_Postprocess,
            'SGS_YoloV2_Postprocess':self.split_SGS_YoloV2_Postprocess,
            'SGS_YoloV3_Postprocess':self.split_SGS_YoloV3_Postprocess,
            'SGS_LanceNet_Postprocess':self.split_SGS_LanceNet_Postprocess,
            'SGS_FDA_Postprocess':self.split_SGS_FDA_Postprocess,
            'PriorBox_RFC':self.split_PriorBox_RFC,
            'QIRQuantize':self.split_QIRQuantize
        }
        self._inputNameShapeMap = inputNameShapeMap
        self._inputName = inputName
        self._SGSModel = model
        self._outputName = outputName
        self._oriModel = copy.deepcopy(model)
        self._maceOpArray = np.array([])
        self.name_shape_map = {}
        self.finetuneArray = []
        self._input_pack_model_arrays = input_pack_model
        self._output_pack_model_arrays = output_pack_model
        self._lstm_num = 0
        self._gru_num = 0
        self._input_name_format_map = input_name_format_map
        self._ConstList = []
        self._MultiConsumerConstList = []
        self._TransposedConstList = []
        self._newConstCreatedList = []
        self._tensorNumCreatedBySgs = -1

    def creatDynamicTensors(self, SGSModel):
        #1.collect all tensor (include constant and virable tensor)
        outputname_list = []
        outputShape_list = []
        #add constant tensor
        for op in SGSModel.op:
          for outputTensor in op.output:
            outputname_list.append(outputTensor)
          for outputShape in op.output_shape:
            outputShape_list.append(outputShape.dims)
          for inputTensor in op.input:
              if self.is_const_tensor(inputTensor):
                  self._ConstList.append(inputTensor)

        #mace_check(len(outputname_list)==len(outputShape_list),
        #"output name num must same as shape num")
        for i in six.moves.range(len(outputname_list)):
          name = outputname_list[i]
          shape = outputShape_list[i]
          self.add_tensor(SGSModel, name, shape, mace_pb2.DT_FLOAT, None)
          self.name_shape_map[outputname_list[i]] = outputShape_list[i]
        for j in six.moves.range(len(self._inputName)):
        #add net input tensor
          name = self._inputName[j]
          shape = self._inputNameShapeMap[name]
          self.add_tensor(SGSModel, name, shape, mace_pb2.DT_FLOAT, None)
          self.name_shape_map[name] = shape
        for tensor in SGSModel.tensors:
          self.name_shape_map[tensor.name] = tensor.dims

    def isMultiConsumerConst(self,constname):
        nameCounter = dict(Counter(self._ConstList))
        self._MultiConsumerConstList = [key for key,value in nameCounter.items()if value > 1]
        if constname in self._MultiConsumerConstList:
            return True
        else:
            return False

    def get_const_ori_data(self, tensor_name):
        if self.find_tensor_by_name(tensor_name).int32_data != []:
            const_input_ori_data = self.find_tensor_by_name(tensor_name).int32_data
        else:
            const_input_ori_data = self.find_tensor_by_name(tensor_name).float_data
        if len(const_input_ori_data) != 0:
            const_data = copy.deepcopy(np.array(const_input_ori_data))
        return const_data

    def OpInputInsertTranspose(self, op):
        # insert NCHW-NHWC transpose for all input tensors of 4dims
        ori_op = copy.deepcopy(op)
        op_name = ori_op.name
        op_input_list = []
        for op_ in self._SGSModel.op:
            if op_ == ori_op:
                for i in six.moves.range(len(ori_op.input)):
                    input_name = ori_op.input[i]
                    # Modify the input tensor of the same name
                    if input_name not in op_input_list:
                        op_input_list.append(input_name)
                        input_name_str = input_name
                    else:
                        input_name_str = input_name + '#' + str(i)

                    input_tensor = self.find_tensor_by_name(input_name)
                    if input_tensor != False:
                        input_shape = self.get_shape_by_name(input_name)

                        if len(input_shape) == 4:
                            if self.is_const_tensor(input_name):
                                self.OpInputInsertTransposeForConst(op_,input_name,input_name_str,input_tensor,input_shape,i)
                            else:
                                op_transpose = self._SGSModel.op.add()
                                op_transpose.name = '#SgsOpInTransW2C_'+ input_name_str + '_' + op_name
                                op_transpose.type = 'TRANSPOSE'
                                shape_tensor_name = op_transpose.name + '_shape'
                                shape_tensor_shape = [len(input_shape)]
                                shape_tensor_data = [0,2,3,1]
                                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                                mace_pb2.DT_INT32, shape_tensor_data)
                                op_transpose.input.extend([input_name])
                                op_transpose.input.extend([shape_tensor_name])
                                #new_input_name = op_transpose.name + '_output'
                                new_input_name = input_name_str + '_#sgsnhwc_' + op_name
                                op_transpose.output.extend([new_input_name])
                                tmp_dim = [0,2,3,1]
                                tmp_shape = [0,1,2,3]
                                for j in six.moves.range(len(tmp_dim)):
                                    tmp_shape[j] = input_shape[tmp_dim[j]]
                                new_input_shape = tmp_shape
                                op_transpose.output_shape.add()
                                op_transpose.output_shape[0].dims.extend(new_input_shape)
                                self.add_tensor(self._SGSModel ,new_input_name, new_input_shape,
                                                input_tensor.data_type, None)
                                self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                                op_.input[i] = new_input_name
                                self.name_shape_map[new_input_name] = new_input_shape

    def OpInputInsertTransposeForConst(self, op, const_name, const_name_str, const_tensor, const_shape,inputnum):
        # Document const-tensor used by multiple consumers,
        # The first appearance keeps the original name,
        # Create new const tensor name if repeated.
        op_ = op
        input_name = const_name
        input_name_str = const_name_str
        input_tensor = const_tensor
        input_shape = const_shape
        i = inputnum
        op_name = op_.name
        data = self.get_const_ori_data(input_name)
        new_data = list(((data.reshape(input_shape)).transpose(0,2,3,1)).flat)
        tmp_dim = [0,2,3,1]
        tmp_shape = [0,1,2,3]
        for j in six.moves.range(len(tmp_dim)):
            tmp_shape[j] = input_shape[tmp_dim[j]]
        new_input_shape = tmp_shape
        if self.isMultiConsumerConst(input_name):
            if input_name not in self._TransposedConstList:
                self._TransposedConstList.append(input_name)
                new_input_name = input_name_str + '_#sgsnhwc_' + op_name
                self.add_tensor(self._SGSModel ,new_input_name, new_input_shape,
                                input_tensor.data_type, new_data)
            else:
                self._TransposedConstList.append(input_name)
                ConstNameCounter = dict(Counter(self._TransposedConstList))
                num = str(ConstNameCounter[input_name]-1)
                new_input_name = input_name_str + num + '_#sgsnhwc_' + op_name
                self.add_tensor(self._SGSModel ,new_input_name, new_input_shape,
                                input_tensor.data_type, new_data)
        else:
            new_input_name = input_name_str
            self.remove_tensor_by_name(input_name)
            self.add_tensor(self._SGSModel ,new_input_name, new_input_shape,
                            input_tensor.data_type, new_data)
        op_.input[i] = new_input_name
        self.name_shape_map[new_input_name] = new_input_shape

    def OpOutputInsertTranspose(self, op):
        # insert NHWC-NCHW transpose for all Output tensors of 4dims
        ori_op = copy.deepcopy(op)
        op_name = ori_op.name
        for op_ in self._SGSModel.op:
            if op_ == ori_op:
                for i in six.moves.range(len(ori_op.output)):
                    output_name = ori_op.output[i]
                    output_tensor = self.find_tensor_by_name(output_name)
                    if output_tensor != False:
                        output_shape = self.get_shape_by_name(output_name)
                        if len(output_shape) == 4:
                            tmp_dim = [0,2,3,1]
                            tmp_shape = [0,1,2,3]
                            for j in six.moves.range(len(tmp_dim)):
                                tmp_shape[j] = output_shape[tmp_dim[j]]
                            new_output_shape = tmp_shape
                            new_output_name = output_name + '_#sgsnhwc_' + op_name
                            self.add_tensor(self._SGSModel ,new_output_name, new_output_shape,
                                            output_tensor.data_type, None)

                            op_transpose = self._SGSModel.op.add()
                            op_transpose.name = '#SgsOpOutTransC2W_'+ output_name + '_' + op_name
                            op_transpose.type = 'TRANSPOSE'
                            shape_tensor_name = op_transpose.name + '_shape'
                            shape_tensor_shape = [len(output_shape)]
                            shape_tensor_data = [0,3,1,2]
                            self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                            mace_pb2.DT_INT32, shape_tensor_data)
                            op_transpose.input.extend([new_output_name])
                            op_transpose.input.extend([shape_tensor_name])
                            op_transpose.output.extend([output_name])
                            op_transpose.output_shape.add()
                            op_transpose.output_shape[0].dims.extend(output_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                            op_.output[i] = new_output_name
                            self.name_shape_map[new_output_name] = new_output_shape

    def ConvertModelInputTensors(self):
        model = copy.deepcopy(self._SGSModel)
        for i in six.moves.range(len(self._inputName)):
            input_tensor = self.find_tensor_by_name(self._inputName[i])
            ori_input_shape = self.get_shape_by_name(self._inputName[i])

            if len(ori_input_shape) == 4:
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = '#SgsModelInTransC2W_' + self._inputName[i]
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0,3,1,2]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([self._inputName[i]])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = self._inputName[i] + '_#sgsnchw'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(ori_input_shape)
                self.add_tensor(self._SGSModel, output_op_transpose, ori_input_shape,
                                mace_pb2.DT_FLOAT, None)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                self.name_shape_map[output_op_transpose] = ori_input_shape[:]
                self._inputNameShapeMap[output_op_transpose] = ori_input_shape[:]

                tmp_dim = [0,2,3,1]
                tmp_shape = [0,1,2,3]
                for k in six.moves.range(len(tmp_dim)):
                    tmp_shape[k] = ori_input_shape[tmp_dim[k]]
                input_tensor.dims[:] = tmp_shape[:]

                for op in model.op:
                    for j in six.moves.range(len(op.input)):
                        if op.input[j] == self._inputName[i]:
                            for op_ in self._SGSModel.op:
                                if op_.name == op.name:
                                    op_.input[j] = output_op_transpose
                                    for ori_op in self._oriModel.op:
                                        if ori_op.name == op_.name:
                                            ori_op.input[j] = output_op_transpose
                                            break
                                    break

    def ConvertModelOutputTensors(self):
        model = copy.deepcopy(self._SGSModel)
        for op_ in model.op:
            oriOpOutput = copy.deepcopy(op_.output)
            for oriOpOutName in oriOpOutput:
                if oriOpOutName in self._outputName:
                    opOutputTensor = self.find_tensor_by_name(oriOpOutName)
                    oriOutputShape = self.get_shape_by_name(oriOpOutName)
                    op_name = op_.name
                    if len(opOutputTensor.dims)==4:
                        # create new op output tensor replacing origin output tensor
                        newOpOutName = oriOpOutName  + '_#sgsnchw'
                        self.add_tensor(self._SGSModel ,newOpOutName, oriOutputShape,
                                        opOutputTensor.data_type, None)
                        self.name_shape_map[newOpOutName] = oriOutputShape[:]
                        index = 0
                        for outputName in op_.output:
                            if outputName != oriOpOutName:
                                index=index+1
                        for op_real in self._SGSModel.op:
                            if op_real == op_:
                                op_real.output[index] = newOpOutName

                        # add transpose before model output for nchw-nhwc
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = '#SgsModelOutTransW2C_' + oriOpOutName
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shape'
                        shape_tensor_data = [0,2,3,1]
                        shape_tensor_shape = [4]
                        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                        mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([newOpOutName])
                        op_transpose.input.extend([shape_tensor_name])
                        op_transpose.output.extend([oriOpOutName])
                        tmp_dim = [0,2,3,1]
                        tmp_shape = [0,1,2,3]
                        for i in six.moves.range(len(tmp_dim)):
                            tmp_shape[i] = oriOutputShape[tmp_dim[i]]
                        op_transpose.output_shape.add()
                        op_transpose.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                        # create new model output tensor shape format
                        opOutputTensor.dims[:] = tmp_shape[:]

                        for other_op in self._SGSModel.op:
                            oriOpInput = copy.deepcopy(other_op.input)
                            for i in six.moves.range(len(oriOpInput)):
                                if oriOpInput[i] == oriOpOutName:
                                    other_op.input[i] = newOpOutName

    def ConvertModelNchwInputTensors(self):
        input_pack_model = []
        if self._input_pack_model_arrays != None and self._input_pack_model_arrays != 'None':
            for i in six.moves.range(len(self._input_pack_model_arrays)):
                if self._input_pack_model_arrays[i] == "NCHW":
                    inputName = copy.deepcopy(self._inputName[i])
                    input_pack_model.append(inputName)

        model = copy.deepcopy(self._SGSModel)
        for i in six.moves.range(len(self._inputName)):
            input_name = self._inputName[i]
            input_tensor = self.find_tensor_by_name(input_name)
            ori_input_shape = self.get_shape_by_name(input_name)

            if input_name in input_pack_model:
                if len(ori_input_shape) == 4:
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = '#SgsModelNchwInTransW2C_' + input_name
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = [0,2,3,1]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                    mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([input_name])
                    op_transpose.input.extend([shape_tensor_name])
                    output_op_transpose = input_name + '_#nhwc'
                    op_transpose.output.extend([output_op_transpose])
                    op_transpose.output_shape.add()
                    op_transpose.output_shape[0].dims.extend(ori_input_shape)
                    self.add_tensor(self._SGSModel, output_op_transpose, ori_input_shape,
                                    mace_pb2.DT_FLOAT, None)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    self.name_shape_map[output_op_transpose] = ori_input_shape[:]

                    tmp_dim = [0,3,1,2]
                    tmp_shape = [0,1,2,3]
                    for k in six.moves.range(len(tmp_dim)):
                        tmp_shape[k] = ori_input_shape[tmp_dim[k]]
                    input_tensor.dims[:] = tmp_shape[:]

                    for op in model.op:
                        for j in six.moves.range(len(op.input)):
                            if op.input[j] == input_name:
                                for op_ in self._SGSModel.op:
                                    if op_.name == op.name:
                                        op_.input[j] = output_op_transpose
                                        for ori_op in self._oriModel.op:
                                            if ori_op.name == op_.name:
                                                ori_op.input[j] = output_op_transpose
                                                break
                                        break

    def ConvertModelNchwOutputTensors(self):
        output_pack_model = []
        if self._output_pack_model_arrays != None and self._output_pack_model_arrays != 'None':
            for i in six.moves.range(len(self._output_pack_model_arrays)):
                if self._output_pack_model_arrays[i] == "NCHW":
                    outputName = copy.deepcopy(self._outputName[i])
                    output_pack_model.append(outputName)

        model = copy.deepcopy(self._SGSModel)
        for op_ in model.op:
            oriOpOutput = copy.deepcopy(op_.output)
            for oriOpOutName in oriOpOutput:
                if oriOpOutName in self._outputName:
                    opOutputTensor = self.find_tensor_by_name(oriOpOutName)
                    oriOutputShape = self.get_shape_by_name(oriOpOutName)
                    op_name = op_.name

                    if oriOpOutName in output_pack_model:
                        # dim=4 output is NCHW, add NHWC-NCHW transpose.
                        if len(opOutputTensor.dims) == 4 and oriOpOutName in output_pack_model:
                            # create new op output tensor replacing origin output tensor
                            newOpOutName = oriOpOutName  + '_#nhwc'
                            self.add_tensor(self._SGSModel ,newOpOutName, oriOutputShape,
                                            opOutputTensor.data_type, None)
                            self.name_shape_map[newOpOutName] = oriOutputShape[:]
                            index = 0
                            for outputName in op_.output:
                                if outputName != oriOpOutName:
                                    index=index+1
                            for op_real in self._SGSModel.op:
                                if op_real == op_:
                                    op_real.output[index] = newOpOutName

                            # add transpose before model output for nchw-nhwc
                            op_transpose = self._SGSModel.op.add()
                            op_transpose.name = '#SgsModelNchwOutTransC2W_' + oriOpOutName
                            op_transpose.type = 'TRANSPOSE'
                            shape_tensor_name = op_transpose.name + '_shape'
                            shape_tensor_data = [0,3,1,2]
                            shape_tensor_shape = [4]
                            self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                            mace_pb2.DT_INT32, shape_tensor_data)
                            op_transpose.input.extend([newOpOutName])
                            op_transpose.input.extend([shape_tensor_name])
                            op_transpose.output.extend([oriOpOutName])
                            tmp_dim = [0,3,1,2]
                            tmp_shape = [0,1,2,3]
                            for i in six.moves.range(len(tmp_dim)):
                                tmp_shape[i] = oriOutputShape[tmp_dim[i]]
                            op_transpose.output_shape.add()
                            op_transpose.output_shape[0].dims.extend(tmp_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                            # create new model output tensor shape format
                            opOutputTensor.dims[:] = tmp_shape[:]

                            for other_op in self._SGSModel.op:
                                oriOpInput = copy.deepcopy(other_op.input)
                                for i in six.moves.range(len(oriOpInput)):
                                    if oriOpInput[i] == oriOpOutName:
                                        other_op.input[i] = newOpOutName

                        # dim<4 output is NCHW, add reshape [3,2,2] to [1,3,2,2].
                        elif len(opOutputTensor.dims) < 4 and oriOpOutName in output_pack_model:
                            # create new op output tensor replacing origin output tensor
                            newOpOutName = oriOpOutName + '_#nhwc'
                            self.add_tensor(self._SGSModel ,newOpOutName, oriOutputShape,
                                            opOutputTensor.data_type, None)
                            self.name_shape_map[newOpOutName] = oriOutputShape[:]
                            index = 0
                            for outputName in op_.output:
                                if outputName != oriOpOutName:
                                    index=index+1
                            for op_real in self._SGSModel.op:
                                if op_real == op_:
                                    op_real.output[index] = newOpOutName

                            # add reshape before model output for converting to 4d NCHW output
                            op_reshape = self._SGSModel.op.add()
                            op_reshape.name = '#SgsModelNchwOutReshape_' + oriOpOutName
                            op_reshape.type = 'RESHAPE'
                            shape_tensor_name = op_reshape.name + '_shape'
                            tmp_shape = [1,1,1,1]
                            output_len = len(oriOutputShape)
                            for i in six.moves.range(output_len):
                                tmp_shape[3-i] = oriOutputShape[output_len-1-i]
                            shape_tensor_data = tmp_shape
                            shape_tensor_shape = [4]
                            self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                            mace_pb2.DT_INT32, shape_tensor_data)
                            op_reshape.input.extend([newOpOutName])
                            op_reshape.input.extend([shape_tensor_name])
                            op_reshape.output.extend([oriOpOutName])
                            op_reshape.output_shape.add()
                            op_reshape.output_shape[0].dims.extend(tmp_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                            # create new model output tensor shape format
                            opOutputTensor.dims[:] = tmp_shape[:]

                            for other_op in self._SGSModel.op:
                                oriOpInput = copy.deepcopy(other_op.input)
                                for i in six.moves.range(len(oriOpInput)):
                                    if oriOpInput[i] == oriOpOutName:
                                        other_op.input[i] = newOpOutName

    def is_const_tensor(self, tensor_name):
        net = self._oriModel
        for tensor in net.tensors:
            if tensor.name == tensor_name:
                return True
        for index,tensor in enumerate(self._newConstCreatedList):
            if tensor == tensor_name:
                return True
        return False

    def is_scalar_tensor(self, tensor_name):
        tensor = self.find_tensor_by_name(tensor_name)
        if len(tensor.dims) == 0:
            return True
        elif len(tensor.dims) == 1 and tensor.dims[0] == 1:
            return True
        else:
            return False

    def run(self):
        self.creatDynamicTensors(self._SGSModel)
        self.ConvertModelInputTensors()

        for op in self._oriModel.op:
          type = op.type
          self._splitOP_transform[type](op)

        self.ConvertModelOutputTensors()

        self.ConvertModelNchwInputTensors()
        self.ConvertModelNchwOutputTensors()
        self.remove_useless_tensor(self._SGSModel)
        return self._SGSModel,self._maceOpArray

    def remove_useless_tensor(self, net):
        tensors_array = copy.deepcopy(net.tensors)
        for tensor in tensors_array:
            remove = True
            for op_ in net.op:
                if tensor.name in op_.input:
                    remove = False
                    break
                if tensor.name in op_.output:
                    remove = False
                    break
            if remove == True:
                net.tensors.remove(tensor)
                #six.print_("Remove tensor: %s" % tensor.name)

    @staticmethod
    def replace(obj_list, source, target):
        for i in six.moves.range(len(obj_list)):
            if obj_list[i] == source:
                obj_list[i] = target

    @staticmethod
    def transpose_shape(shape, order):
        transposed_shape = []
        for i in six.moves.range(len(order)):
            transposed_shape.append(shape[order[i]])
        shape[:] = transposed_shape[:]

    @staticmethod
    def normalize_op_name(name):
        return name.replace(':', '_')

    def get_tensor_shape(self, tensor):
        producer = self._producer[tensor]
        for i in six.moves.range(len(producer.output)):
            if producer.output[i] == tensor:
                return list(producer.output_shape[i].dims)

    def consumer_count(self, tensor_name):
        return len(self._consumers.get(tensor_name, []))

    def is_op_output_node(self, op):
        output_node_tensor_names = [out for out in
                                    self._option.output_nodes]
        for output in op.output:
            if output in output_node_tensor_names:
                return True

        return False
    def finetuneNet(self):
        if len(self.finetuneArray) != 0:
           for map in self.finetuneArray:
             # find op input name == topX
               # top num = len(map) - bottom_num = len(map) - 1
               bottom_name = map['bottom']
               for i in six.moves.range(len(map)-1):
                 split_top_name = map['top'+str(i)]
                 isOutputNode = True
                 for op_ in self._SGSModel.op:
                   for j,bottom in enumerate(op_.input):
                     if split_top_name == bottom:
                       op_.input[j] =  bottom_name
                       isOutputNode = False
                       self.remove_tensor_by_name(split_top_name)

    def safe_remove_node(self, op, replace_op=None, remove_input_tensor=False):
        """remove op.
        1. change the inputs of its consumers to the outputs of replace_op
        2. if the op is output node, change output node to replace op"""

        if replace_op is None:
            # When no replace op specified, we change the inputs of
            # its consumers to the input of the op. This handles the case
            # that the op is identity op and its input is a tensor.
            mace_check(len(op.output) == 1 and len(op.input) == 1,
                       "cannot remove op that w/o replace op specified"
                       " and input/output length > 1\n" + str(op))

            for consumer_op in self._consumers.get(op.output[0], []):
                self.replace(consumer_op.input, op.output[0], op.input[0])

            mace_check(op.output[0] not in self._option.output_nodes,
                       "cannot remove op that is output node")
        else:
            mace_check(len(op.output) == len(replace_op.output),
                       "cannot remove op since len(op.output) "
                       "!= len(replace_op.output)")

            for i in six.moves.range(len(op.output)):
                for consumer_op in self._consumers.get(op.output[i], []):
                    self.replace(consumer_op.input,
                                 op.output[i],
                                 replace_op.output[i])

            # if the op is output node, change replace_op output name to the op
            # output name
            for i in six.moves.range(len(op.output)):
                if op.output[i] in self._option.output_nodes:
                    for consumer in self._consumers.get(
                            replace_op.output[i], []):
                        self.replace(consumer.input,
                                     replace_op.output[i],
                                     op.output[i])
                    replace_op.output[i] = op.output[i]

        if remove_input_tensor:
            for input_name in op.input:
                if input_name in self._consts:
                    const_tensor = self._consts[input_name]
                    self._model.tensors.remove(const_tensor)

        self._model.op.remove(op)

    def add_in_out_tensor_info(self):
        net = self._model
        for input_node in self._option.input_nodes.values():
            input_info = net.input_info.add()
            input_info.name = input_node.name
            input_info.data_format = input_node.data_format.value
            input_info.dims.extend(input_node.shape)
            input_info.data_type = mace_pb2.DT_FLOAT

        output_nodes = self._option.check_nodes.values()
        for output_node in output_nodes:
            output_info = net.output_info.add()
            output_info.name = output_node.name
            output_info.data_format = output_node.data_format.value
            output_info.dims.extend(
                self._producer[output_node.name].output_shape[0].dims)
            output_info.data_type = mace_pb2.DT_FLOAT

        return False

    def remove_identity_op(self):
        net = self._SGSModel
        for op in net.op:
            if op.type == 'Identity':
                #six.print_("Remove identity: %s(%s)" % (op.name, op.type))
                self.safe_remove_node(op,
                                      self._producer.get(op.input[0], None))
                return True

        return False

    def remove_op_with_name(self,name):
        net = self._SGSModel
        for op in net.op:
            if op.name == name:
                #six.print_("Remove op: %s(%s)" % (op.name, op.type))
                self._SGSModel.op.remove(op)
                return True

        return False
    def add_tensor(self, net, name, shape, data_type, value ,data_fromat = mace_pb2.DT_NCHW):
        #check dumplicated name
        for tensor in net.tensors:
            if tensor.name == name:
                print("find dumplicated tensor name",name)
                assert 0
        tensor = net.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_format = data_fromat
        tensor.data_type = data_type
        if data_type == mace_pb2.DT_INT32:
            tensor.int32_data.extend(value)
        else:
            tensor.float_data.extend(value)
        return tensor
    def find_tensor_by_name(self, name):
        net = self._SGSModel
        for tensor in net.tensors:
               if tensor.name == name:
                   return tensor
        six.print_("can not find tensor: %s" % (name))
        return False
    def remove_tensor_by_name(self, name):
        net = self._SGSModel
        for tensor in net.tensors:
               if tensor.name == name:
                   net.tensors.remove(tensor)
        #six.print_("can not find tensor: %s" % (name))
        return False
    def get_shape_by_name(self, name):
        try:
          shape = self.name_shape_map[name]
        except Exception:
          six.print_("can't find name in map",name)
          assert 0
        return shape

    def get_axes_by_shape(self, axis, dims, NCHW=True, convert_negative=True):
        if convert_negative:
            axis = dims + axis if axis < 0 else axis
        axis_result = axis
        if dims == 4 and NCHW:
            if axis == 0:
                axis_result = 0
            if axis == 1:
                axis_result = 3
            elif axis == 2:
                axis_result = 1
            elif axis == 3:
                axis_result = 2
        return axis_result

    def handle_const(self, op, op_type, const_name, const_tensor, const_shape,index = 1):
        def reshape_const(op, const_name,const_tensor,const_shape,data,const_index):
            # reshep != 4dim to 4dim
            if len(const_shape) != 4 and self.is_scalar_tensor(const_name) == 0:
                num = 4 - len(const_shape)
                for i in range(num):
                    const_shape.insert(0,1)
                data = data.reshape(const_shape)
                data = list(data.flat)
                # create new const tensor
                const_tensor_name = const_name + "_" + op.name + '_reshape'
                const_tensor_shape = const_shape
                const_tensor_data = data
                self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                const_tensor.data_type, const_tensor_data)
                op.input[const_index] = const_tensor_name
                self._newConstCreatedList.append(const_tensor_name)
                self.name_shape_map[const_tensor_name] = const_tensor_shape
                del const_tensor
                return op

        def transpose_const(op,const_name,const_tensor,const_shape,data,const_index,tmp_dim,tmp_shape,create_tensor = True):
            new_data = list(((data.reshape(const_shape)).transpose(tmp_dim)).flat)
            for i in six.moves.range(len(tmp_dim)):
                tmp_shape[i] = const_shape[tmp_dim[i]]
            # for i in six.moves.range(len(tmp_dim)):
            #     const_tensor.dims[i] = tmp_shape[i]
            if create_tensor:
                const_tensor_name = const_name + '_' + op.name + '_SGSconst'
                const_tensor_shape = tmp_shape
                const_tensor_data = new_data
                self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                mace_pb2.DT_FLOAT, const_tensor_data)
                op.input[const_index] = const_tensor_name
                self._newConstCreatedList.append(const_tensor_name)
                del const_tensor
                return op
            else:
                return tmp_shape,const_tensor,new_data

        # main
        const_index = index
        data = self.get_const_ori_data(const_name)
        if op_type == "PRELU":
            transpose_dim = [1,2,0]
            ori_dim = [0,1,2]
            op = transpose_const(op,const_name,const_tensor,const_shape,data,const_index,transpose_dim,ori_dim)
            return
        elif op_type == "DECONV_2D":
            transpose_dim = [1,2,3,0]
            ori_dim = [0,1,2,3]
            op = transpose_const(op,const_name,const_tensor,const_shape,data,const_index,transpose_dim,ori_dim)
            return
        elif op_type == "CONV_2D":
            transpose_dim = [0,2,3,1]
            ori_dim = [0,1,2,3]
            op = transpose_const(op,const_name,const_tensor,const_shape,data,const_index,transpose_dim,ori_dim)
            return
        elif op_type == "DECONV_3D":
            transpose_dim = [2,3,4,0,1]
            ori_dim = [0,1,2,3,4]
            tmp_shape,const_tensor,new_data = transpose_const(op,const_name,const_tensor,const_shape,data,const_index,transpose_dim,ori_dim,create_tensor = False)
            return tmp_shape,const_tensor,new_data
        elif op_type == "CONV_3D":
            transpose_dim = [2,3,4,1,0]
            ori_dim = [0,1,2,3,4]
            tmp_shape,const_tensor,new_data = transpose_const(op,const_name,const_tensor,const_shape,data,const_index,transpose_dim,ori_dim,create_tensor = False)
            return tmp_shape,const_tensor,new_data
        elif op_type == "ELTWISE":
            reshape_const(op,const_name,const_tensor,const_shape,data,const_index)
            return
        elif op_type == "MATMUL":
            if len(const_shape) == 2:
                transpose_dim = [1,0]
                ori_dim = [0,1]
            elif len(const_shape) == 3:
                transpose_dim = [0,2,1]
                ori_dim = [0,1,2]
            elif len(const_shape) == 4:
                transpose_dim = [0,1,3,2]
                ori_dim = [0,1,2,3]
            op = transpose_const(op,const_name,const_tensor,const_shape,data,const_index,transpose_dim,ori_dim)
            return
        else:
            mace_check(False,"Constants of this type are not supported!")

    def is_tensor_detached(self, name):
        is_detached = True
        for op in self._oriModel.op:
            for input_name in op.input:
                if name == input_name:
                    is_detached = False
        for model_output_name in self._outputName:
            if name == model_output_name:
                is_detached = False
        return is_detached

    def cast_tensor(self, Tensor, Dtype):
        if Tensor.data_type != Dtype:
            if self.is_const_tensor(Tensor.name):
                if Dtype == mace_pb2.DT_INT32:
                    Tensor.data_type = mace_pb2.DT_INT32
                    Tensor.int32_data.extend(np.array(copy.deepcopy(Tensor.float_data)).astype(np.int32).tolist())
                elif Dtype == mace_pb2.DT_FLOAT:
                    Tensor.data_type = mace_pb2.DT_FLOAT
                    Tensor.float_data.extend(np.array(copy.deepcopy(Tensor.int32_data)).astype(np.float32).tolist())
                else:
                    mace_check(False, 'Not support data type!')

    def check_NCHW_And_Change(self, input_name, op_name):
        output_shape = self.get_shape_by_name(input_name)
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = 'SGS_' + op_name + '_transpose'
        op_transpose.type = 'TRANSPOSE'
        shape_tensor_name = op_transpose.name + '_shape'
        shape_tensor_data = [0,2,3,1]
        shape_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
            mace_pb2.DT_INT32, shape_tensor_data)
        op_transpose.input.extend([input_name])
        op_transpose.input.extend([shape_tensor_name])
        output_op_transpose = op_transpose.name + '_output'
        op_transpose.output.extend([output_op_transpose])
        '''
        tmp_dim = [0,2,3,1]# NCHW --[0312]--> NWCH  --[0231]--> NCHW
        tmp_shape = [0,1,2,3]
        for i in six.moves.range(len(tmp_dim)):
          tmp_shape[i] = output_shape[tmp_dim[i]]
        op_transpose.output_shape.add()
        op_transpose.output_shape[0].dims.extend(tmp_shape)
        '''
        op_transpose.output_shape.add()
        op_transpose.output_shape[0].dims.extend(output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_transpose)
        return output_op_transpose

    def CreateNewIniFor3D(self,path,input_name_array):
        copy_num = len(input_name_array)
        config = configparser.ConfigParser()
        # auto generate autogen_input_config.ini
        autogen_path = getAutogenIniPath(path, config)
        # change 'input_config.ini' content
        for key in config['INPUT_CONFIG']:
            if key == 'inputs':
                inputs_value = ','.join(input_name_array) + ';'
                config.set('INPUT_CONFIG','inputs',inputs_value)

            elif key == 'input_formats':
                input_formats_value = config['INPUT_CONFIG']['input_formats'].replace(" ","")
                input_formats_value = input_formats_value.strip(string.punctuation)
                input_formats_result = ''
                for i in six.moves.range(copy_num):
                    input_formats_result = input_formats_result + input_formats_value + ','
                input_formats_result = input_formats_result.rstrip(',')
                config.set('INPUT_CONFIG','input_formats',input_formats_result)

            elif key == 'training_input_formats':
                training_input_formats_value = config['INPUT_CONFIG']['training_input_formats'].replace(" ","")
                training_input_formats_value = training_input_formats_value.strip(string.punctuation)
                training_input_formats_result = ''
                for i in six.moves.range(copy_num):
                    training_input_formats_result = training_input_formats_result + training_input_formats_value + ','
                training_input_formats_result = training_input_formats_result.rstrip(',')
                config.set('INPUT_CONFIG','training_input_formats',training_input_formats_result)

            elif key == 'quantizations':
                quantizations_value = config['INPUT_CONFIG']['quantizations'].replace(" ","")
                quantizations_value = quantizations_value.strip(string.punctuation)
                quantizations_result = ''
                for i in six.moves.range(copy_num):
                    quantizations_result = quantizations_result + quantizations_value + ','
                quantizations_result = quantizations_result.rstrip(',')
                config.set('INPUT_CONFIG','quantizations',quantizations_result)

            elif key == 'mean':
                mean_value = config['INPUT_CONFIG']['mean'].replace(" ","")
                mean_value = mean_value.strip(string.punctuation)
                mean_result = ''
                for i in six.moves.range(copy_num):
                    mean_result = mean_result + mean_value + ','
                mean_result = mean_result.rstrip(',')
                config.set('INPUT_CONFIG','mean', mean_result)

            elif key == 'mean_red':
                mean_red_value = config['INPUT_CONFIG']['mean_red'].replace(" ","")
                mean_red_value = mean_red_value.strip(string.punctuation)
                mean_red_result = ''
                for i in six.moves.range(copy_num):
                    mean_red_result = mean_red_result + mean_red_value + ','
                mean_red_result = mean_red_result.rstrip(',')
                config.set('INPUT_CONFIG','mean_red',mean_red_result)

            elif key == 'mean_green':
                mean_green_value = config['INPUT_CONFIG']['mean_green'].replace(" ","")
                mean_green_value = mean_green_value.strip(string.punctuation)
                mean_green_result = ''
                for i in six.moves.range(copy_num):
                    mean_green_result = mean_green_result + mean_green_value + ','
                mean_green_result = mean_green_result.rstrip(',')
                config.set('INPUT_CONFIG','mean_green',mean_green_result)

            elif key == 'mean_blue':
                mean_blue_value = config['INPUT_CONFIG']['mean_blue'].replace(" ","")
                mean_blue_value = mean_blue_value.strip(string.punctuation)
                mean_blue_result = ''
                for i in six.moves.range(copy_num):
                    mean_blue_result = mean_blue_result + mean_blue_value + ','
                mean_blue_result = mean_blue_result.rstrip(',')
                config.set('INPUT_CONFIG','mean_blue',mean_blue_result)

            elif key == 'std_value':
                std_value = config['INPUT_CONFIG']['std_value'].replace(" ","")
                std_value = std_value.strip(string.punctuation)
                std_value_result = ''
                for i in six.moves.range(copy_num):
                   std_value_result = std_value_result + std_value + ','
                std_value_result = std_value_result.rstrip(',')
                config.set('INPUT_CONFIG','std_value',std_value_result)

        with open(autogen_path, mode='w', encoding='utf-8', errors='ignore') as f:
            config.write(f)
        setAutogenWarning(autogen_path)

    def split_Activation(self, op):
        # Transposed
        leakyrelu_coefficient,isLeakRelu,isPRelu = 0,0,0
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_activation_type_str:
                        a_type = arg[i].s
                        if a_type.decode() == 'RELU':
                            self.OpInputInsertTranspose(op_)
                            op_.type = 'RELU'
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)
                        if a_type.decode() == 'LEAKYRELU':
                            self.OpInputInsertTranspose(op_)
                            op_.type = "LEAKY_RELU"
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)
                        if a_type.decode() == 'PRELU':
                            self.OpInputInsertTranspose(op_)
                            op_.type = "PRELU"
                            input1 = op_.input[1]
                            input1_tensor = self.find_tensor_by_name(input1)
                            input1_shape = self.get_shape_by_name(input1)
                            if len(input1_shape) == 3:
                                if self.is_const_tensor(input1) == True:
                                    self.handle_const(op_,op_.type,input1,input1_tensor,input1_shape)
                                elif self.is_const_tensor(input1) == False:
                                    op_transpose = self._SGSModel.op.add()
                                    op_transpose.name = 'SGS_' + op_name + '_transpose'
                                    op_transpose.type = 'TRANSPOSE'
                                    shape_tensor_name = op_transpose.name + '_shape'
                                    shape_tensor_data = [1,2,0]
                                    shape_tensor_shape = [3]
                                    self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                                    op_transpose.input.extend([input1])
                                    op_transpose.input.extend([shape_tensor_name])
                                    output_op_transpose = op_transpose.name + '_output'
                                    op_transpose.output.extend([output_op_transpose])
                                    tmp_shape = [input1_shape[1],input1_shape[2],input1_shape[0]]
                                    op_transpose.output_shape.add()
                                    op_transpose.output_shape[0].dims.extend(tmp_shape)
                                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                                    op_.input[1] = output_op_transpose
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)
                        if a_type.decode() == 'SIGMOID':
                            self.OpInputInsertTranspose(op_)
                            op_.type = "LOGISTIC"
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)
                        if a_type.decode() == 'RELUX':
                            self.OpInputInsertTranspose(op_)
                            op_.type = "RELU6"
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)
                        if a_type.decode() =='TANH':
                            self.OpInputInsertTranspose(op_)
                            op_.type = "TANH"
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)

    def split_ArgMax(self, op):
        # Transposed according keepdim
        for op_ in self._SGSModel.op:
            if op_ == op:
                arg = op_.arg
                keepdim = 1
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_keepdims_str: # "keepdims"
                        keepdim = arg[i].i
                axis = 0
                output_shape = op_.output_shape[:]
                if keepdim:
                    self.OpInputInsertTranspose(op_)
                    inputShape = self.get_shape_by_name(op_.input[0])
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            axis = self.get_axes_by_shape (arg[i].i, len(inputShape), True, True)
                    op_.type = "ARG_MAX"
                    #add axis tensor into input arrays
                    axis_tensor_name = op_.name + '_axis'
                    axis_data = [axis]
                    axis_data_shape = [1]
                    self.add_tensor(self._SGSModel, axis_tensor_name, axis_data_shape,
                                    mace_pb2.DT_INT32, axis_data)
                    op_.input.extend([axis_tensor_name])
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)
                else:
                    inputShape = self.get_shape_by_name(op_.input[0])
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            axis = self.get_axes_by_shape (arg[i].i, len(inputShape), False, True)
                    op_.type = "ARG_MAX"
                    #add axis tensor into input arrays
                    axis_tensor_name = op_.name + '_axis'
                    axis_data = [axis]
                    axis_data_shape = [1]
                    self.add_tensor(self._SGSModel, axis_tensor_name, axis_data_shape,
                                    mace_pb2.DT_INT32, axis_data)
                    op_.input.extend([axis_tensor_name])
                    self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_BatchNorm(self, op):
        # Transposed
        # yi = xi * scale_value + offset_value
        # output_shape == input_shape
        for op_ in self._SGSModel.op:
           if op_ == op:
                self.OpInputInsertTranspose(op_)
                name = op_.name
                xi = op_.input[0]
                scale_value = op_.input[1]
                offset_value = op_.input[2]
                outputName = op_.output[0]
                input_shape = self.get_shape_by_name(xi)
                inputTensor = self.find_tensor_by_name(xi)
                add_transpose = False
                new_shape = copy.deepcopy(input_shape)
                #  input shape is N,C,D1,D2,...
                #  change to N,D1,D2,...,C
                if len(input_shape) != 4:
                    #transpor + mul + add + transpose
                    #cal new_shape
                    #new_shape = copy.deepcopy(input_shape)
                    new_shape[-1] = input_shape[1]
                    for i in six.moves.range(len(input_shape)-2):
                        new_shape[1+i] = input_shape[2+i]
                    #cal new_shape order
                    new_shape_order = [i for i in six.moves.range(len(input_shape))]
                    new_shape_temp = copy.deepcopy(new_shape_order)
                    new_shape_order[-1] = new_shape_temp[1]
                    for i in six.moves.range(len(new_shape_temp)-2):
                        new_shape_order[1+i] = new_shape_temp[2+i]
                    # creat transpose op
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = 'SGS_' + name + '_transpose'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = new_shape_order
                    shape_tensor_shape = [len(new_shape_order)]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                        mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([xi])
                    op_transpose.input.extend([shape_tensor_name])
                    output_op_transpose = op_transpose.name + '_output'
                    op_transpose.output.extend([output_op_transpose])
                    op_transpose.output_shape.add()
                    op_transpose.output_shape[0].dims.extend(new_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    add_transpose = True

                out_shape = new_shape

                #creat Mul
                op_mul = self._SGSModel.op.add()
                op_mul.name = name + '_MUL'
                op_mul.type = 'MUL'
                if add_transpose:
                    op_mul.input.extend([output_op_transpose])
                else:
                    op_mul.input.extend([xi])
                op_mul.input.extend([scale_value])
                output_name_mul = op_mul.name + '_output'
                op_mul.output.extend([output_name_mul])
                op_mul.output_shape.add()
                op_mul.output_shape[0].dims.extend(out_shape)
                #tensor data is variable,so didn't creat new tensor
                self._maceOpArray = np.append(self._maceOpArray,op_mul)

                #creat ADD
                op_add = self._SGSModel.op.add()
                op_add.name = name + '_ADD'
                op_add.type = 'ADD'
                op_add.input.extend([output_name_mul])
                op_add.input.extend([offset_value])
                if add_transpose:
                    output_name_add = op_add.name + '_output'
                    op_add.output.extend([output_name_add])
                    op_add.output_shape.add()
                    op_add.output_shape[0].dims.extend(out_shape)
                else:
                    op_add.output[:] = op_.output[:]
                    op_add.output_shape.add()
                    op_add.output_shape[0].dims.extend(out_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_add)

                if add_transpose:
                    #cal out shape order
                    out_shape_order = [i for i in six.moves.range(len(input_shape))]
                    out_shape_order[1] = new_shape_temp[-1]
                    for i in six.moves.range(len(new_shape_temp)-2):
                        out_shape_order[2+i] = new_shape_temp[1+i]
                    # creat transpose op
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = 'SGS_' + name + '_transpose#2'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = out_shape_order
                    shape_tensor_shape = [len(out_shape_order)]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                        mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([output_name_add])
                    op_transpose.input.extend([shape_tensor_name])
                    op_transpose.output[:] = op_.output[:]
                    op_transpose.output_shape.extend(op_.output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                else:
                    self.OpOutputInsertTranspose(op_add)

                #remove BatchNorm op
                self.remove_op_with_name(name)

    def split_Cast(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CAST"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Concat(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
          if op_ == op:
            arg = op_.arg
            inputShape = self.get_shape_by_name(op_.input[0])
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_axis_str:
                    arg[i].i = self.get_axes_by_shape (arg[i].i, len(inputShape), True, True)
            self.OpInputInsertTranspose(op_)
            op_.type = "CONCATENATION"
            self._maceOpArray = np.append(self._maceOpArray,op_)
            self.OpOutputInsertTranspose(op_)

    def split_Conv1D(self, op):
        op_name = op.name
        if "\'" in op_name:
            op_name = op_name[2:-2]
        x = op.input[0]
        shape = self.get_shape_by_name(x)
        w = self.find_tensor_by_name(op.input[1]).float_data

        # create  transpose op
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = 'SGS_' + op_name + '#transpose'
        op_transpose.type = 'TRANSPOSE'
        op_transpose.input.extend([x])
        transpose_output_tensor_name = op_transpose.name + '_output_shape'
        transpose_output_tensor_data = [0, 2, 1]  # n,w,c
        transpose_output_tensor_shape = [3]
        self.add_tensor(self._SGSModel, transpose_output_tensor_name, transpose_output_tensor_shape,
                        mace_pb2.DT_INT32, transpose_output_tensor_data)
        op_transpose.input.extend([transpose_output_tensor_name])
        output_op_transpose = op_transpose.name + '_output'
        op_transpose.output.extend([output_op_transpose])
        op_transpose.output_shape.add()
        op_transpose.output_shape[0].dims.extend([shape[0], shape[2], shape[1]])
        self._maceOpArray = np.append(self._maceOpArray, op_transpose)

        # creat reshape op
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '#reshape'
        op_reshape.type = 'RESHAPE'
        op_reshape.input.extend([output_op_transpose])
        reshape_output_tensor_name = op_reshape.name + '_output_shape'
        reshape_output_tensor_data = [op_transpose.output_shape[0].dims[0], 1, op_transpose.output_shape[0].dims[1],
                                      op_transpose.output_shape[0].dims[2]]  # n,h,w,c
        reshape_output_tensor_shape = [4]
        self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
        op_reshape.input.extend([reshape_output_tensor_name])
        output_op_reshape = op_reshape.name + '_output'
        op_reshape.output.extend([output_op_reshape])
        op_reshape.output_shape.add()
        op_reshape.output_shape[0].dims.extend([op_transpose.output_shape[0].dims[0],1, op_transpose.output_shape[0].dims[1],
                                                 op_transpose.output_shape[0].dims[2]])
        self._maceOpArray = np.append(self._maceOpArray, op_reshape)

        # creat conv op
        arg = op.arg
        weight = op.input[1] # has been changed to 4dim in mace
        bias = op.input[2]
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == MaceKeyword.mace_padding_values_str:
                # padding value should be divided by 2
                paddingL, paddingR, paddingT, paddingB = arg[i].ints
            elif name == MaceKeyword.mace_strides_str:
                strideH, strideW = arg[i].ints

        # transpose 4dim weight
        weight_shape = self.get_shape_by_name(weight)
        weight_tensor = self.find_tensor_by_name(weight)
        if self.is_const_tensor(weight) == True:
            self.handle_const(op,"CONV_2D",weight,weight_tensor,weight_shape)
        op_conv = self._SGSModel.op.add()
        op_conv.name = op_name + '_conv#1'
        op_conv.type = 'CONV_2D'
        paddingType_arg = op_conv.arg.add()
        paddingType_arg.name = MaceKeyword.mace_padding_str
        paddingType_arg.i = self.pooling_paddingType['CAFFE']
        strides_arg = op_conv.arg.add()
        strides_arg.name = 'strides'
        strides_arg.ints.extend([strideH, strideW])
        padding_arg = op_conv.arg.add()
        padding_arg.name = 'padding_values'
        padding_arg.ints.extend([paddingL, paddingR, paddingT, paddingB])
        op_conv.input.extend([output_op_reshape])
        op_conv.input.extend([op.input[1]])
        op_conv.input.extend([bias])
        output_op_conv = op_conv.name + '_output'
        op_conv.output.extend([output_op_conv])
        op_conv.output_shape.add()
        op_conv.output_shape[0].dims.extend([op.output_shape[0].dims[0],1, op.output_shape[0].dims[2],
                                              op.output_shape[0].dims[1]])  # nchw
        self._maceOpArray = np.append(self._maceOpArray, op_conv)

        # creat reshape op
        op_reshape2 = self._SGSModel.op.add()
        op_reshape2.name = op_name + '_reshape2'
        op_reshape2.type = 'RESHAPE'
        reshape2_output_tensor_name = op_reshape2.name + '_output_shape'
        reshape2_output_tensor_data = [op_conv.output_shape[0].dims[0], op_conv.output_shape[0].dims[2],
                                       op_conv.output_shape[0].dims[3]]
        reshape2_output_tensor_shape = [3]
        self.add_tensor(self._SGSModel, reshape2_output_tensor_name, reshape2_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape2_output_tensor_data)
        op_reshape2.input.extend([output_op_conv])
        op_reshape2.input.extend([reshape2_output_tensor_name])
        output_op_reshape2 = op_reshape2.name + '_output'
        op_reshape2.output.extend([output_op_reshape2])
        op_reshape2.output_shape.add()
        op_reshape2.output_shape[0].dims.extend([op_conv.output_shape[0].dims[0], op_conv.output_shape[0].dims[2],
                                                 op_conv.output_shape[0].dims[3]])  # nchw
        self._maceOpArray = np.append(self._maceOpArray, op_reshape2)

        # create  transpose op
        op_transpose2 = self._SGSModel.op.add()
        op_transpose2.name = op_name + '#transpose2'
        op_transpose2.type = 'TRANSPOSE'
        op_transpose2.input.extend([output_op_reshape2])
        transpose2_output_tensor_name = op_transpose2.name + '_output_shape'
        transpose2_output_tensor_data = [0, 2, 1]  # n,c,w,h
        transpose2_output_tensor_shape = [3]
        self.add_tensor(self._SGSModel, transpose2_output_tensor_name, transpose2_output_tensor_shape,
                        mace_pb2.DT_INT32, transpose2_output_tensor_data)
        op_transpose2.input.extend([transpose2_output_tensor_name])
        op_transpose2.output[:] = op.output[:]
        op_transpose2.output_shape.add()
        op_transpose2.output_shape[0].dims.extend([op_reshape2.output_shape[0].dims[0], op_reshape2.output_shape[0].dims[2],
                                                   op_reshape2.output_shape[0].dims[1]])
        self._maceOpArray = np.append(self._maceOpArray, op_transpose2)

        #self.remove_op_with_name(op_name)

    def split_Conv2D(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                xi = op_.input[0]
                arg = op_.arg
                op_name = op_.name
                strideH,strideW = 0,0
                dilationH,dilationW = 1,1
                group = 1
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                    #padding value should be divided by 2
                        paddingL,paddingR,paddingT,paddingB = arg[i].ints
                    elif name == MaceKeyword.mace_strides_str:
                        strideH,strideW = arg[i].ints
                    elif name == MaceKeyword.mace_dilations_str:
                        dilationH,dilationW = arg[i].ints
                    elif name == MaceKeyword.mace_kernel_str:
                        kernelH,kernelW = arg[i].ints
                    elif name == "group":
                        group = arg[i].i

                if group == 1:
                    #check pad value
                    [ni,hi,wi,ci] = self.get_shape_by_name(xi)
                    [no,co,ho,wo] = op_.output_shape[0].dims
                    kernel_dilationH =  (kernelH - 1) * dilationH + 1
                    kernel_dilationW =  (kernelW - 1) * dilationW + 1
                    ibH = (ho - 1) * strideH + kernel_dilationH
                    ibW = (wo - 1) * strideW + kernel_dilationW
                    need_check = False
                    if ibW < wi + paddingL + paddingR:
                        need_check = True
                        paddingR = ibW - (wi + paddingL)
                        if paddingR < 0:
                            paddingR = 0
                    if ibH < hi + paddingT + paddingB:
                        need_check = True
                        paddingB = ibH - (hi + paddingT)
                        if paddingB < 0:
                            paddingB = 0
                    if need_check:
                        for i in six.moves.range(len(arg)):
                            name = arg[i].name
                            if name == MaceKeyword.mace_padding_values_str:
                            #padding value should be divided by 2
                                arg[i].ints[:] = paddingL,paddingR,paddingT,paddingB
                    op_.type = "CONV_2D"
                    #add padding_type
                    paddingType_arg = arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    #set padding_type
                    paddingType_arg.i = self.pooling_paddingType['CAFFE']
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)
                else:
                    op_.name = 'GroupConv' + op_name
                    op_.type = 'CUSTOM'
                    #add padding_type
                    paddingType_arg = arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    #set padding_type
                    paddingType_arg.i = self.pooling_paddingType['CAFFE']
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)

    def split_Conv3D(self, op):
      # 5 dimensions follow the following rules:
      # 1. input transpose (NCDHW-> NDHWC)
      #    (1)  if the first input operator of the model is CONV3D:
      #            do not use the method of inserting the TRAN operator for conversion; (NCDHW-> NDHWC)
      #         if input dataformat is non-RAWDATA:
      #            split input according to D dim into 4dims input shapes (NCDHW-> NHWC * Dnum)
      #    (2)  if CONV3D is not the first node of the model:
      #            convert by inserting TRAN operator;(NCDHW-> NDHWC)
      # 2. weight transpose (NCDHW ->DHWNC)
      #    (1)  if const tensor:
      #            do not use the method of inserting the TRAN operator for conversion; (NCDHW ->DHWNC)
      #    (2)  if variable tensor:
      #            convert by inserting TRAN operator; (NCDHW ->DHWNC)
      # 3. output transpose (NDHWC-> NCDHW)
      # 4. Split group-conv into ordinary conv according to group num
      #    (1)  split input[c]
      #    (2)  split weight[N]
      #    (3)  If group is the first operator of the model,
      #         it must be the case where D does not need to be split.
      for op_ in self._SGSModel.op:
          if op_ == op:
              xi = op_.input[0]   # input -> X
              wi = op_.input[1]   # weight -> W
              bias = op_.input[2] # bias->B
              arg = op_.arg
              op_name = op_.name
              strideD,strideH,strideW = 1,1,1
              dilationD,dilationH,dilationW = 1,1,1
              group = 1
              for i in six.moves.range(len(arg)):
                  name = arg[i].name
                  if name == MaceKeyword.mace_padding_values_str:
                      paddingI,paddingL,paddingT,paddingO,paddingR,paddingB = arg[i].ints
                  elif name == MaceKeyword.mace_strides_str:
                      strideD,strideH,strideW = arg[i].ints
                  elif name == MaceKeyword.mace_dilations_str:
                      dilationD,dilationH,dilationW = arg[i].ints
                  elif name == 'group':
                      group = arg[i].i

              inputTensor = self.find_tensor_by_name(xi)
              inputShape = self.get_shape_by_name(xi)
              weightTensor = self.find_tensor_by_name(wi)
              weightShape = self.get_shape_by_name(wi)
              tensor_bias = self.find_tensor_by_name(bias)
              tensor_bias_shape = tensor_bias.dims
              output_shape = op.output_shape[0].dims

              # weight transpose  NCDHW ->DHWNC
              ## transpose const tensor
              if self.is_const_tensor(wi) == True:
                    weightShape_new, weightTensor_new, weight_new_data = self.handle_const(op_, "CONV_3D", wi, weightTensor, weightShape)
              ## transpose unconst tensor
              else:
                  if len(weightShape) == 5:
                      op_transpose0 = self._SGSModel.op.add()
                      op_transpose0.name = op_name + '_transpose'
                      op_transpose0.type = 'TRANSPOSE'
                      shape_tensor_name = op_transpose0.name + '_shape'
                      shape_tensor_data = [2,3,4,1,0]
                      shape_tensor_shape = [5]
                      self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                      op_transpose0.input.extend([wi])
                      op_transpose0.input.extend([shape_tensor_name])
                      output_op_transpose0 = op_transpose0.name + '_output'
                      op_transpose0.output.extend([output_op_transpose0])
                      tmp_dim = [2,3,4,1,0]
                      tmp_shape = [0,1,2,3,4]
                      for i in six.moves.range(len(tmp_dim)):
                          tmp_shape[i] = inputShape[tmp_dim[i]]
                      op_transpose0.output_shape.add()
                      weightShape_new = tmp_shape
                      weightTensor_new = output_op_transpose0
                      self._maceOpArray = np.append(self._maceOpArray,op_transpose0)

              # input transpose NCDHW-> NDHWC
              if len(inputShape) == 5:
                name = xi
                inputShapeChanged = False
                inputShapeSliced = False
                input_tensor_name_array = []
                model_input_num = len(self._inputName)
                # first op is conv3D
                for i in six.moves.range(model_input_num):
                    if name == self._inputName[i] and \
                       not ('RAWDATA_S16_NHWC' == self._input_name_format_map[name] or \
                            'RAWDATA_F32_NHWC' == self._input_name_format_map[name]):
                        input_ori_shape = self._inputNameShapeMap[name]
                        n = input_ori_shape[0]
                        c = input_ori_shape[1]
                        d = input_ori_shape[2]
                        h = input_ori_shape[3]
                        w = input_ori_shape[4]
                        tmp_dim = [0,2,3,4,1]
                        tmp_shape = [0,1,2,3,4]
                        for j in six.moves.range(len(tmp_dim)):
                          tmp_shape[j] = input_ori_shape[tmp_dim[j]]
                          inputTensor.dims[j] = tmp_shape[j]
                         # input is unrawdata, slice D
                        if not ('RAWDATA_S16_NHWC' == self._input_name_format_map[name] or \
                            'RAWDATA_F32_NHWC' == self._input_name_format_map[name]):
                          tmp_shape[1] = 1
                          inputTensor.dims[1] = tmp_shape[1]
                        inputShape = tmp_shape
                        self._inputNameShapeMap[name] = inputShape[:]
                        self.name_shape_map[name] = inputShape[:]
                        #  ori_shape has been reshaped
                        inputShapeChanged = True
                        if not ('RAWDATA_S16_NHWC' == self._input_name_format_map[name] or \
                            'RAWDATA_F32_NHWC' == self._input_name_format_map[name]):
                            keep_input = False # del ori_input
                            for op_other in self._SGSModel.op:
                                if op_other != op_:
                                    for op_other_input in op_other.input:
                                        if op_other_input == op_.input[0]:
                                            keep_input = True
                            inputTensor = self.find_tensor_by_name(xi)
                            if keep_input == False:
                                self.remove_tensor_by_name(xi)

                            for k in range(d):
                                input_tensor_name = self._inputName[i] + "_slice" + str(k)
                                input_tensor_shape = [1,1,1,1]
                                # keep NHWC
                                input_tensor_shape[0] = inputShape[0]
                                input_tensor_shape[1] = inputShape[2]
                                input_tensor_shape[2] = inputShape[3]
                                input_tensor_shape[3] = inputShape[4]
                                # NHWC will be transposed

                                self.add_tensor(self._SGSModel ,input_tensor_name, input_tensor_shape,
                                                mace_pb2.DT_FLOAT, None)
                                input_tensor_name_array.extend([input_tensor_name])
                                self._inputName.extend([input_tensor_name])
                                self._inputNameShapeMap[input_tensor_name] = input_tensor_shape[:]
                                self.name_shape_map[input_tensor_name] = input_tensor_shape[:]
                            self._inputName.remove(self._inputName[0])
                            inputShapeSliced = True
                            new_ini = self.CreateNewIniFor3D(INPUT_CONFIG_INI,self._inputName)
                        break

                # create transpose1->input transpose
                if inputShapeChanged == False:
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = 'SGS_' + op_name + '_transpose1'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape1'
                    shape_tensor_data = [0,2,3,4,1]
                    shape_tensor_shape = [5]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([xi])
                    op_transpose.input.extend([shape_tensor_name])
                    output_op_transpose = op_transpose.name + '_output1'
                    op_transpose.output.extend([output_op_transpose])
                    # transpose inputshape
                    tmp_dim = [0,2,3,4,1]
                    tmp_shape = [0,1,2,3,4]
                    for i in six.moves.range(len(tmp_dim)):
                      tmp_shape[i] = inputShape[tmp_dim[i]]
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(tmp_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                # add  group conv3d
                if group != 1:
                    if inputShapeChanged == False and inputShapeSliced == False:
                        mace_check(weightShape[0] % group == 0,
                                        "tensor weight shape 0 must be multiples of group.")
                        mace_check(inputShape[1] % group == 0,
                                        "tensor weight shape 0 must be multiples of group.")
                        # get weight after TRANS
                        tensor_weight = weightTensor_new
                        tensor_weight_shape = weightShape_new

                        # creat split
                        op_split = self._SGSModel.op.add()
                        op_split.name = op_name + '_split'
                        op_split.type = 'SPLIT'
                        axis_tensor_name = op_split.name + '_axis'
                        axis_data = [4] # [N,D,H,W,C], split axis[4]->split[c]
                        axis_shape = [1]
                        self.add_tensor(self._SGSModel, axis_tensor_name, axis_shape, mace_pb2.DT_INT32, axis_data)
                        split_output_shape = copy.deepcopy(op_transpose_shape)
                        split_output_shape[4] = int(op_transpose_shape[4] / group)
                        op_split.input.extend([axis_tensor_name])
                        op_split.input.extend([output_op_transpose])
                        for i in six.moves.range(group):
                            split_topName = op_split.name + '_output#' + str(i)
                            op_split.output.extend([split_topName])
                            op_split.output_shape.add()
                            op_split.output_shape[i].dims.extend(split_output_shape)
                        numSplits_arg = op_split.arg.add()
                        numSplits_arg.name = MaceKeyword.mace_num_split_str
                        numSplits_arg.i = len(op_split.output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_split)

                        # creat conv3d
                        output_op_conv3d_name_array = np.array([])
                        for i in six.moves.range(group):
                            op_conv3d = self._SGSModel.op.add()
                            op_conv3d.name = op_name + '_conv#' + str(i)
                            op_conv3d.type = 'CONV_3D'
                            # add arg
                            strides_arg = op_conv3d.arg.add()
                            strides_arg.name = 'strides'
                            strides_arg.ints.extend([strideD,strideH,strideW])
                            padding_arg = op_conv3d.arg.add()
                            padding_arg.name = 'padding_values'
                            padding_arg.ints.extend([paddingI,paddingL,paddingT,paddingO,paddingR,paddingB])
                            dilations_arg = op_conv3d.arg.add()
                            dilations_arg.name = 'dilations'
                            dilations_arg.ints.extend([dilationD,dilationH,dilationW])
                            #add padding_type
                            paddingType_arg = op_conv3d.arg.add()
                            paddingType_arg.name = MaceKeyword.mace_padding_str
                            #set padding_type
                            paddingType_arg.i = self.pooling_paddingType['CAFFE']
                            #creat filter
                            filter_tensor_name = op_conv3d.name + "_filter#" + str(i)
                            filter_tensor_shape = copy.deepcopy(weightShape)
                            filter_tensor_shape[0] = int(weightShape[0] / group)
                            offset = len(weight_ori_data) / group
                            filter_tesnor_value = weight_ori_data[int(i*offset):int((i+1)*offset)]
                            data = np.array(filter_tesnor_value)
                            data = data.reshape(filter_tensor_shape)
                            data = data.transpose(2,3,4,1,0)
                            data = list(data.flat)
                            filter_tesnor_value = copy.deepcopy(data)
                            tmp_dim = [2,3,4,1,0]
                            tmp_shape = [0,1,2,3,4]
                            for j in six.moves.range(len(tmp_dim)):
                                tmp_shape[j] = filter_tensor_shape[tmp_dim[j]]
                            filter_tensor_shape = tmp_shape
                            self.add_tensor(self._SGSModel ,filter_tensor_name, filter_tensor_shape, mace_pb2.DT_FLOAT, filter_tesnor_value)

                            #creat bias
                            bias_tensor_name = op_conv3d.name + "_bias#" + str(i)
                            bias_tensor_shape = copy.deepcopy(tensor_bias_shape)
                            bias_tensor_shape[0] = int(tensor_bias_shape[0] / group)
                            offset = len(tensor_bias.float_data) / group
                            bias_tesnor_value = tensor_bias.float_data[int(i*offset):int((i+1)*offset)]
                            self.add_tensor(self._SGSModel ,bias_tensor_name, bias_tensor_shape, mace_pb2.DT_FLOAT, bias_tesnor_value)

                            # creat input W X B
                            op_conv3d.input.extend([op_split.name + '_output#' + str(i)])
                            op_conv3d.input.extend([filter_tensor_name])
                            op_conv3d.input.extend([bias_tensor_name])

                            # creat output
                            output_op_conv3d_name = op_conv3d.name + '_output#' + str(i)
                            output_op_conv3d_name_array = np.append(output_op_conv3d_name_array,output_op_conv3d_name)
                            op_conv3d.output.extend([output_op_conv3d_name])
                            op_conv3d.output_shape.add()
                            tmp_dim = [0,2,3,4,1]
                            tmp_shape2 = [0,1,2,3,4]
                            for i in six.moves.range(len(tmp_dim)):
                                tmp_shape2[i] = output_shape[tmp_dim[i]]
                            conv3d_output_shape = tmp_shape2
                            conv3d_output_shape[4] = int(conv3d_output_shape[4] / group)
                            op_conv3d.output_shape[0].dims.extend(conv3d_output_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_conv3d)

                        # create concat
                        tmp_dim = [0,2,3,4,1]
                        tmp_shape3 = [0,1,2,3,4]
                        for i in six.moves.range(len(tmp_dim)):
                            tmp_shape3[i] = output_shape[tmp_dim[i]]
                        concat_shape = tmp_shape3
                        op_concat = self._SGSModel.op.add()
                        op_concat.name = op_name + '_concat'
                        op_concat.type = 'CONCATENATION'
                        axis_arg = op_concat.arg.add()
                        axis_arg.name = 'axis'
                        axis_arg.i = 4
                        for name in output_op_conv3d_name_array:
                            op_concat.input.extend([name])
                        output_op_concat = op_concat.name + '_output'
                        op_concat.output.extend([output_op_concat])
                        op_concat.output_shape.add()
                        op_concat.output_shape[0].dims.extend(concat_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_concat)
                        del tensor_weight
                        del tensor_bias
                        self.remove_op_with_name(op_name)

                      # creat transpose from NCHW back to NHWC
                        op_transpose3 = self._SGSModel.op.add()
                        op_transpose3.name = op_name + '_transpose3'
                        op_transpose3.type = 'TRANSPOSE'
                        shape_tensor_name3 = op_transpose3.name + '_shape3'
                        shape_tensor_data3 = [0,4,1,2,3]
                        shape_tensor_shape = [5]
                        self.add_tensor(self._SGSModel ,shape_tensor_name3, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data3)
                        op_transpose3.input.extend([output_op_concat])
                        op_transpose3.input.extend([shape_tensor_name3])
                        op_transpose3.output.extend(op.output)
                        op_transpose3.output_shape.add()
                        op_transpose3.output_shape[0].dims.extend(output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose3)

                # add conv3d
                else:
                    # ADD conv3d
                    op_conv3d= self._SGSModel.op.add()
                    op_conv3d.name = op_name + '_conv3d'
                    op_conv3d.type = 'CONV_3D'

                    #create arg
                    strides_arg = op_conv3d.arg.add()
                    strides_arg.name = 'strides'
                    strides_arg.ints.extend([strideD,strideH,strideW])
                    padding_arg = op_conv3d.arg.add()
                    padding_arg.name = 'padding_values'
                    padding_arg.ints.extend([paddingI,paddingL,paddingT,paddingO,paddingR,paddingB])
                    dilations_arg = op_conv3d.arg.add()
                    dilations_arg.name = 'dilations'
                    dilations_arg.ints.extend([dilationD,dilationH,dilationW])
                    #add padding_type
                    paddingType_arg = op_conv3d.arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    #set padding_type
                    paddingType_arg.i = self.pooling_paddingType['CAFFE']

                    # create filter
                    filter_tensor_name = op_conv3d.name + '_filter'
                    filter_tensor_shape = copy.deepcopy(weightShape_new)
                    filter_tesnor_value = weight_new_data
                    self.add_tensor(self._SGSModel ,filter_tensor_name, filter_tensor_shape ,mace_pb2.DT_FLOAT, filter_tesnor_value)

                    # create bias
                    bias_tensor_name = op_conv3d.name + '_bias'
                    bias_tensor_shape = copy.deepcopy(tensor_bias_shape)
                    bias_tesnor_value = tensor_bias.float_data
                    self.add_tensor(self._SGSModel ,bias_tensor_name, bias_tensor_shape ,mace_pb2.DT_FLOAT, bias_tesnor_value)

                    # create input W  X  B
                    ## X
                    if inputShapeChanged == True and inputShapeSliced == True:
                        op_conv3d.input.extend(input_tensor_name_array)
                    elif inputShapeChanged == True and inputShapeSliced == False:
                        op_conv3d.input.extend([xi])
                    else:
                        op_conv3d.input.extend([output_op_transpose])
                    ## W
                    if self.is_const_tensor(wi) == True:
                        op_conv3d.input.extend([filter_tensor_name])
                    else:
                        op_conv3d.input.extend([output_op_transpose0])
                    ## B
                    op_conv3d.input.extend([bias_tensor_name])

                    #create output
                    output_op_conv3d = op_conv3d.name + '_output'
                    op_conv3d.output.extend([output_op_conv3d])
                    op_conv3d.output_shape.add()
                    tmp_dim = [0,2,3,4,1]
                    tmp_shape = [0,1,2,3,4]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = output_shape[tmp_dim[i]]
                    conv3d_output_shape = tmp_shape
                    op_conv3d.output_shape[0].dims.extend(conv3d_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_conv3d)

                    # creat transpose from NCHW back to NHWC
                    op_transpose3 = self._SGSModel.op.add()
                    op_transpose3.name = op_name + '_transpose3'
                    op_transpose3.type = 'TRANSPOSE'
                    shape_tensor_name3 = op_transpose3.name + '_shape3'
                    shape_tensor_data3 = [0,4,1,2,3]
                    shape_tensor_shape = [5]
                    self.add_tensor(self._SGSModel ,shape_tensor_name3, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data3)
                    op_transpose3.input.extend([output_op_conv3d])
                    op_transpose3.input.extend([shape_tensor_name3])
                    op_transpose3.output.extend(op.output)
                    op_transpose3.output_shape.add()
                    op_transpose3.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose3)

    def split_CReLU(self, op):
        # unTransposed
        # (mul * -1 + leakRelu)    leakRelu
        #           |              |
        #           | - - - -|- - -|
        #                    |
        #                  concat
        #   ONNX nonexistent operator
        op_name = op.name
        xi = op.input[0]
        [n,c,h,w] = self.get_shape_by_name(xi)
        slope,axis = 0.0,1
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'negative_slope':
            slope= arg[i].f
          elif name == 'concat_axis':
            axis = arg[i].i
        # creat mul
        op_mul = self._SGSModel.op.add()
        op_mul.name = name + '_MUL'
        op_mul.type = 'MUL'
        scale_value_tensor_name = op_mul.name + '_scale'
        scale_value = -1
        scale_data = [scale_value]
        scale_shape = [1]
        self.add_tensor(self._SGSModel ,scale_value_tensor_name, scale_shape,
            mace_pb2.DT_FLOAT, scale_data)
        op_mul.input.extend([xi])
        op_mul.input.extend([scale_value_tensor_name])
        output_name_mul = op_mul.name + '_output'
        op_mul.output.extend([output_name_mul])
        op_mul.output_shape.add()
        op_mul.output_shape[0].dims.extend([n,c,h,w])
        self._maceOpArray = np.append(self._maceOpArray,op_mul)
        # creat leakyRelu
        op_lkyRelu = self._SGSModel.op.add()
        op_lkyRelu.name = op_name + '_lkyRelu'
        op_lkyRelu.type = 'LEAKY_RELU'
        slope_arg = op_lkyRelu.arg.add()
        slope_arg.name = MaceKeyword.mace_activation_leakyrelu_coefficient_str
        slope_arg.f = slope
        op_lkyRelu.input.extend([output_name_mul])
        output_op_lkyRelu = op_lkyRelu.name + '_output'
        op_lkyRelu.output.extend([output_op_lkyRelu])
        op_lkyRelu.output_shape.add()
        op_lkyRelu.output_shape[0].dims.extend([n,c,h,w])
        self._maceOpArray = np.append(self._maceOpArray,op_lkyRelu)
        # creat leakyRelu
        op_lkyRelu = self._SGSModel.op.add()
        op_lkyRelu.name = op_name + '_lkyRelu' + "#1"
        op_lkyRelu.type = 'LEAKY_RELU'
        slope_arg = op_lkyRelu.arg.add()
        slope_arg.name = MaceKeyword.mace_activation_leakyrelu_coefficient_str
        slope_arg.f = slope
        op_lkyRelu.input.extend([xi])
        output_op_lkyRelu_1 = op_lkyRelu.name + '_output'
        op_lkyRelu.output.extend([output_op_lkyRelu_1])
        op_lkyRelu.output_shape.add()
        op_lkyRelu.output_shape[0].dims.extend([n,c,h,w])
        self._maceOpArray = np.append(self._maceOpArray,op_lkyRelu)
        # creat concat
        op_concat = self._SGSModel.op.add()
        op_concat.name = op_name + '_concat'
        op_concat.type = 'CONCATENATION'
        axis_arg = op_concat.arg.add()
        axis_arg.name = 'axis'
        axis_arg.i = axis
        op_concat.input.extend([output_op_lkyRelu_1])
        op_concat.input.extend([output_op_lkyRelu])
        output_op_concat = op_concat.name + '_output'
        op_concat.output[:] = op.output[:]
        op_concat.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_concat)
        self.remove_op_with_name(op_name)


    def split_Crop(self, op):
        # unTransposed
        # nonexistent ONNX operator
        op_name = op.name
        input_data_name = op.input[0]
        [n,c,h,w] = op.output_shape[0].dims[:]
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'axis':
            axis= arg[i].i
          elif name == 'offset':
            offset = arg[i].ints
        begin_slice = [0,0,0,0]
        offset_list = [0,0,0,0]
        if len(offset) == 1:
          offset_list = [offset[0],offset[0],offset[0],offset[0]]
        else:
          offset_list = offset
        for i in six.moves.range(len(begin_slice)):
          if i >= axis:
            begin_slice[i] = offset_list[i-axis]
        [bn,bc,bh,bw] =  begin_slice[:]
        # creat slice
        op_slice = self._SGSModel.op.add()
        op_slice.name = op_name + '_slice'
        op_slice.type = 'SLICE'
        begin_tensor_name = op_slice.name + '_begin'
        begin_tensor_data = [bn,bh,bw,bc]
        begin_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,begin_tensor_name, begin_tensor_shape,
            mace_pb2.DT_INT32, begin_tensor_data)
        size_tensor_name = op_slice.name + '_size'
        size_tensor_data = [n,h,w,c]
        size_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,size_tensor_name, size_tensor_shape,
            mace_pb2.DT_INT32, size_tensor_data)
        op_slice.input.extend([input_data_name])
        op_slice.input.extend([begin_tensor_name])
        op_slice.input.extend([size_tensor_name])
        op_slice.output[:] = op.output[:]
        op_slice.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_slice)
        self.remove_op_with_name(op_name)

    def split_ConvTranspose3D(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                # get args
                op_name = op.name
                strideD,strideW,strideH = 0,0,0
                dilationD,dilationW,dilationH = 1,1,1
                arg = op.arg
                group = 0
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                        paddingI,paddingL,paddingT,paddingO,paddingR,paddingB= arg[i].ints
                    elif name == MaceKeyword.mace_strides_str:
                        strideD,strideH,strideW = arg[i].ints
                    elif name == MaceKeyword.mace_dilations_str:
                        dilationD,dilationH,dilationW = arg[i].ints
                    elif name == 'group':
                        group = arg[i].i
                biase_name = op.input[2]
                tensor_bias = self.find_tensor_by_name(biase_name)
                tensor_bias_shape = tensor_bias.dims
                # 1\get [Ni, Ci, Di,Hi, Wi] from input
                ## input: [Ni, Ci,Di, Hi, Wi]
                xi = op.input[0]
                inputTensor = self.find_tensor_by_name(xi)
                [ni,ci,di,hi,wi] = input_shape = self.get_shape_by_name(xi)

                # 2\get [No, Co, Ho, Wo] from output
                ## output: [No, Co, Ho, Wo]
                [n,c,d,h,w] = output_shape = op.output_shape[0].dims[:]

                # 3\get [Nk, Ck, Dk, Hk, Wk] from  kernel weight
                ## weight: [in_channels, out_channels//groups, kernel_size[0], kernel_size[1]]
                ## The DI of SGS = in_channels,  The DO of SGS = out_channels,
                ## The DO of SGS cannot be represented by the C-dimensional data of weight, because C-dimensional data of weight equals to The DO of SGS//groups
                wi = op.input[1]
                filter_tensor = self.find_tensor_by_name(wi)
                [nk,ck,dk,hk,wk] = filter_tensor.dims[:]
                di = nk
                do = ck*group
                filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(nk,ck,dk,hk,wk)
                # reverse kernel
                filter_data = filter_data[:, :, ::-1, ::-1,::-1]
                filter_tensor.float_data[:] = filter_data.flat

                #creat transpose for input tensor
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '_transpose1'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape1'
                shape_tensor_data = [0,2,3,4,1]
                shape_tensor_shape = [5]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([xi])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output1'
                op_transpose.output.extend([output_op_transpose])
                # transpose inputshape
                tmp_dim = [0,2,3,4,1]
                tmp_shape = [0,1,2,3,4]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = input_shape[tmp_dim[i]]
                op_transpose.output_shape.add()
                op_transpose_shape = copy.deepcopy(tmp_shape)
                op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                def computeOutSidePadding(input_shape,output_shape,strideD,strideH,strideW,kernel_shape):
                    [ni,ci,di,hi,wi] = input_shape
                    [n,c,d,h,w] = output_shape
                    input_pading_shape_D = di + (di - 1)*(strideD - 1)
                    input_pading_shape_H = hi + (hi - 1)*(strideH - 1)
                    input_pading_shape_W = wi + (wi - 1)*(strideW - 1)
                    round_func = math.floor
                    pad_d = round_func((d - 1 - input_pading_shape_D + kernel_shape[0])/2)
                    pad_h = round_func((h - 1 - input_pading_shape_H + kernel_shape[1])/2)
                    pad_w = round_func((w - 1 - input_pading_shape_W + kernel_shape[2])/2)
                    dilation_output_d = input_pading_shape_D + 2 * pad_d
                    dilation_output_h = input_pading_shape_H + 2 * pad_h
                    dilation_output_w = input_pading_shape_W + 2 * pad_w
                    return pad_d,pad_h,pad_w,dilation_output_d,dilation_output_h,dilation_output_w

                pad_d,pad_h,pad_w,dilation_output_d,dilation_output_h,dilation_output_w = computeOutSidePadding(input_shape,output_shape,strideD,strideH,strideW,filter_tensor.dims[2:])
                #                #
                # create dilation #
                #                #
                #NCDHW ->NDHWC
                op_dilation = self._SGSModel.op.add()
                op_dilation.name = 'Dilation' + op_name
                op_dilation.type = 'CUSTOM'
                # 1.add outside pad tensor
                outside_pad_name = op_.name + '_dilation' + '_outside_pad'
                outside_pad_data = [0,0,pad_d,pad_d,pad_h,pad_h,pad_w,pad_w,0,0] #[[N][D][H][W][C]]
                outside_pad_shape = [5,2]
                self.add_tensor(self._SGSModel ,outside_pad_name, outside_pad_shape,
                    mace_pb2.DT_INT32, outside_pad_data)
                # 2.add inside pad tensor
                inside_pad_name = op_.name + '_dilation' + '_inside_pad'
                inside_pad_data = [0,int(strideD-1),int(strideH-1),int(strideW-1),0]
                inside_pad_shape = [5]
                self.add_tensor(self._SGSModel ,inside_pad_name, inside_pad_shape,
                    mace_pb2.DT_INT32, inside_pad_data)
                op_dilation.input.extend([output_op_transpose])
                op_dilation.input.extend([outside_pad_name])
                op_dilation.input.extend([inside_pad_name])
                output_name_dilation = op_.name + '_dilation' + '_output'
                op_dilation.output.extend([output_name_dilation])
                op_dilation.output_shape.add()
                op_dilation.output_shape[0].dims.extend([ni,dilation_output_d,dilation_output_h,dilation_output_w,ci])
                self._maceOpArray = np.append(self._maceOpArray,op_dilation)
                #                #
                # create conv3d   #
                #                #

                # weight transpose  NCDHW ->DHWCN
                ## transpose const tensor
                weightTensor = self.find_tensor_by_name(wi)
                weightShape = self.get_shape_by_name(wi)
                if self.is_const_tensor(wi) == True:
                    weightShape_new, weightTensor_new, weight_new_data = self.handle_const(op_, "DECONV_3D", wi, weightTensor, weightShape)
                # ADD conv3d
                op_conv3d= self._SGSModel.op.add()
                op_conv3d.name = op_name + '_conv3d'
                op_conv3d.type = 'CONV_3D'

                #create arg
                strides_arg = op_conv3d.arg.add()
                strides_arg.name = 'strides'
                strideD,strideW,strideH = 1,1,1
                strides_arg.ints.extend([strideD,strideH,strideW])
                padding_arg = op_conv3d.arg.add()
                padding_arg.name = 'padding_values'
                padding_arg.ints.extend([0,0,0,0,0,0])
                dilations_arg = op_conv3d.arg.add()
                dilations_arg.name = 'dilations'
                dilations_arg.ints.extend([dilationD,dilationH,dilationW])
                #add padding_type
                paddingType_arg = op_conv3d.arg.add()
                paddingType_arg.name = MaceKeyword.mace_padding_str
                #set padding_type
                paddingType_arg.i = self.pooling_paddingType['CAFFE']

                # create filter
                filter_tensor_name = op_conv3d.name + '_filter'
                filter_tensor_shape = copy.deepcopy(weightShape_new)
                filter_tesnor_value = weight_new_data
                self.add_tensor(self._SGSModel ,filter_tensor_name, filter_tensor_shape ,mace_pb2.DT_FLOAT, filter_tesnor_value)

                # create bias
                bias_tensor_name = op_conv3d.name + '_bias'
                bias_tensor_shape = copy.deepcopy(tensor_bias_shape)
                bias_tesnor_value = tensor_bias.float_data
                self.add_tensor(self._SGSModel ,bias_tensor_name, bias_tensor_shape ,mace_pb2.DT_FLOAT, bias_tesnor_value)

                # create input W  X  B
                ## X
                op_conv3d.input.extend([output_name_dilation])
                ## W
                op_conv3d.input.extend([filter_tensor_name])
                ## B
                op_conv3d.input.extend([bias_tensor_name])

                #create output
                output_op_conv3d = op_conv3d.name + '_output'
                op_conv3d.output.extend([output_op_conv3d])
                op_conv3d.output_shape.add()
                tmp_dim = [0,2,3,4,1]
                tmp_shape = [0,1,2,3,4]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = output_shape[tmp_dim[i]]
                conv3d_output_shape = tmp_shape
                op_conv3d.output_shape[0].dims.extend(conv3d_output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_conv3d)

                # creat transpose from NCHW back to NHWC
                op_transpose3 = self._SGSModel.op.add()
                op_transpose3.name = op_name + '_transpose3'
                op_transpose3.type = 'TRANSPOSE'
                shape_tensor_name3 = op_transpose3.name + '_shape3'
                shape_tensor_data3 = [0,4,1,2,3]
                shape_tensor_shape = [5]
                self.add_tensor(self._SGSModel ,shape_tensor_name3, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data3)
                op_transpose3.input.extend([output_op_conv3d])
                op_transpose3.input.extend([shape_tensor_name3])
                op_transpose3.output.extend(op.output)
                op_transpose3.output_shape.add()
                op_transpose3.output_shape[0].dims.extend(output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose3)

    def split_ConvTranspose1D(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                # get args
                op_name = op.name
                strideW,strideH = 0,0
                dilationW,dilationH = 1,1
                arg = op.arg
                group = 0
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                        paddingL,paddingT,paddingR,paddingB= arg[i].ints
                    elif name == MaceKeyword.mace_strides_str:
                        strideH,strideW = arg[i].ints
                    elif name == MaceKeyword.mace_dilations_str:
                        dilationH,dilationW = arg[i].ints
                    elif name == 'group':
                        group = arg[i].i

                biase_name = op.input[2]
                tensor_bias = self.find_tensor_by_name(biase_name)
                tensor_bias_shape = tensor_bias.dims
                # 1\get [Ni, Ci, Wi] from input
                ## input: [Ni, Ci,Wi]
                xi = op.input[0]
                inputTensor = self.find_tensor_by_name(xi)
                [ni,ci,wi] = input_shape = self.get_shape_by_name(xi)

                # 2\get [No, Co, Wo] from output
                ## output: [No, Co, Wo]
                [no,co,wo] = output_shape = op.output_shape[0].dims[:]

                # 3\get [Nk, Ck, Wk] from  kernel weight
                ## weight: [in_channels, out_channels//groups, kernel_size[0], kernel_size[1]]
                ## The DI of SGS = in_channels,  The DO of SGS = out_channels,
                ## The DO of SGS cannot be represented by the C-dimensional data of weight, because C-dimensional data of weight equals to The DO of SGS//groups
                weight = op.input[1]
                filter_tensor = self.find_tensor_by_name(weight)
                [nk,ck,hk,wk] = filter_tensor.dims[:]
                di = nk
                do = ck*group
                filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(nk,ck,hk,wk)
                # reverse kernel
                filter_data = filter_data[:, :, ::-1,::-1]
                filter_tensor.float_data[:] = filter_data.flat

                # create  transpose op
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '#transpose'
                op_transpose.type = 'TRANSPOSE'
                op_transpose.input.extend([xi])
                # set transpose's output_shape
                transpose_output_tensor_name = op_transpose.name + '_output_shape'
                transpose_output_tensor_data = [0, 2, 1]  # n,w,c
                transpose_output_tensor_shape = [3]
                self.add_tensor(self._SGSModel, transpose_output_tensor_name, transpose_output_tensor_shape,
                                mace_pb2.DT_INT32, transpose_output_tensor_data)
                op_transpose.input.extend([transpose_output_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend([ni,wi,ci])
                self._maceOpArray = np.append(self._maceOpArray, op_transpose)

                # creat reshape op
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '#reshape'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_transpose])
                # set reshape's output_shape
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [ni, 1, wi, ci]  # n,h,w,c
                reshape_output_tensor_shape = [4]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend([ni, 1, wi, ci])
                self._maceOpArray = np.append(self._maceOpArray, op_reshape)


                def computeOutSidePadding(input_shape,output_shape,strideH,strideW,kernel_w_shape):
                    [ni,ci,wi] = input_shape
                    [n,c,w] = output_shape
                    hi, h, kernel_h_shape = 1,1,1
                    input_pading_shape_H = hi + (hi - 1)*(strideH - 1)
                    input_pading_shape_W = wi + (wi - 1)*(strideW - 1)
                    round_func = math.floor
                    pad_h = round_func((h - 1 - input_pading_shape_H + kernel_h_shape)/2)
                    pad_w = round_func((w - 1 - input_pading_shape_W + kernel_w_shape)/2)
                    dilation_output_h = input_pading_shape_H + 2 * pad_h
                    dilation_output_w = input_pading_shape_W + 2 * pad_w
                    return pad_h,pad_w,dilation_output_h,dilation_output_w

                pad_h,pad_w,dilation_output_h,dilation_output_w = computeOutSidePadding(input_shape,output_shape,strideH,strideW,filter_tensor.dims[-1])
                #                #
                # create dilation #
                #                #
                op_dilation = self._SGSModel.op.add()
                op_dilation.name = 'Dilation' + op_name
                op_dilation.type = 'CUSTOM'
                # 1.add outside pad tensor
                outside_pad_name = op_.name + '_dilation' + '_outside_pad'
                outside_pad_data = [0,0,pad_h,pad_h,pad_w,pad_w,0,0] #[[n],[h],[w],[c]]
                outside_pad_shape = [4,2]
                self.add_tensor(self._SGSModel ,outside_pad_name, outside_pad_shape,
                    mace_pb2.DT_INT32, outside_pad_data)
                # 2.add inside pad tensor
                inside_pad_name = op_.name + '_dilation' + '_inside_pad'
                inside_pad_data = [0,int(strideH-1),int(strideW-1),0]
                inside_pad_shape = [4]
                self.add_tensor(self._SGSModel ,inside_pad_name, inside_pad_shape,
                    mace_pb2.DT_INT32, inside_pad_data)
                op_dilation.input.extend([output_op_reshape])
                op_dilation.input.extend([outside_pad_name])
                op_dilation.input.extend([inside_pad_name])
                output_name_dilation = op_.name + '_dilation' + '_output'
                op_dilation.output.extend([output_name_dilation])
                op_dilation.output_shape.add()
                op_dilation.output_shape[0].dims.extend([ni,dilation_output_h,dilation_output_w,ci])
                self._maceOpArray = np.append(self._maceOpArray,op_dilation)
                #                #
                # create conv2d   #
                #                #
                op_conv = self._SGSModel.op.add()
                op_conv.name = op_name + '_conv'
                op_conv.type = 'CONV_2D'
                strides_arg = op_conv.arg.add()
                strides_arg.name = 'strides'
                strides_arg.ints.extend([1,1])
                padding_arg = op_conv.arg.add()
                padding_arg.name = 'padding_values'
                padding_arg.ints.extend([0,0,0,0])
                #add padding_type
                paddingType_arg = op_conv.arg.add()
                paddingType_arg.name = MaceKeyword.mace_padding_str
                #set padding_type
                paddingType_arg.i = self.pooling_paddingType['CAFFE']
                op_conv.input.extend([output_name_dilation])
                depth_filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(nk,ck,hk,wk)
                depth_filter_data = np.transpose(depth_filter_data,(1,2,3,0))
                filter_tensor.float_data[:] = depth_filter_data.flat
                filter_tensor.dims[:] = [ck,hk,wk,nk]
                op_conv.input.extend([weight])
                op_conv.input.extend([biase_name])
                output_op_conv = op_.name + '_conv' + '_output'
                op_conv.output.extend([output_op_conv])
                op_conv.output_shape.add()
                op_conv.output_shape[0].dims.extend([no,1,wo,co])
                self._maceOpArray = np.append(self._maceOpArray,op_conv)

                # creat reshape op
                op_reshape2 = self._SGSModel.op.add()
                op_reshape2.name = op_name + '_reshape2'
                op_reshape2.type = 'RESHAPE'
                reshape2_output_tensor_name = op_reshape2.name + '_output_shape'
                reshape2_output_tensor_data = [no, wo, co]
                reshape2_output_tensor_shape = [3]
                self.add_tensor(self._SGSModel, reshape2_output_tensor_name, reshape2_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape2_output_tensor_data)
                op_reshape2.input.extend([output_op_conv])
                op_reshape2.input.extend([reshape2_output_tensor_name])
                output_op_reshape2 = op_reshape2.name + '_output'
                op_reshape2.output.extend([output_op_reshape2])
                op_reshape2.output_shape.add()
                op_reshape2.output_shape[0].dims.extend([no, wo, co])  # nchw
                self._maceOpArray = np.append(self._maceOpArray, op_reshape2)
                # self.remove_op_with_name(op_name)

                # create  transpose op
                op_transpose2 = self._SGSModel.op.add()
                op_transpose2.name = op_name + '#transpose2'
                op_transpose2.type = 'TRANSPOSE'
                op_transpose2.input.extend([output_op_reshape2])
                # set transpose's output_shape
                transpose2_output_tensor_name = op_transpose2.name + '_output_shape'
                transpose2_output_tensor_data = [0, 2, 1]  # n,c,w,h
                transpose2_output_tensor_shape = [3]
                self.add_tensor(self._SGSModel, transpose2_output_tensor_name, transpose2_output_tensor_shape,
                                mace_pb2.DT_INT32, transpose2_output_tensor_data)
                op_transpose2.input.extend([transpose2_output_tensor_name])
                op_transpose2.output[:] = op.output[:]
                op_transpose2.output_shape.add()
                op_transpose2.output_shape[0].dims.extend([no, co, wo])
                self._maceOpArray = np.append(self._maceOpArray, op_transpose2)


    def split_Deconv2D(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                # get args
                op_name = op.name
                strideW,strideH = 0,0
                dilationW,dilationH = 1,1
                output_paddingH,output_paddingW = 0,0
                arg = op.arg
                group = 0
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                        paddingL,paddingR,paddingT,paddingB= arg[i].ints
                    elif name == MaceKeyword.mace_strides_str:
                        strideH,strideW = arg[i].ints
                    elif name == MaceKeyword.mace_dilations_str:
                        dilationH,dilationW = arg[i].ints
                    elif name == 'group':
                        group = arg[i].i
                    elif name == 'output_padding':
                        output_paddingH,output_paddingW = arg[i].ints
                biase_name = op.input[2]
                # 1\get [Ni, Ci, Hi, Wi] from input
                ## input: [Ni, Ci, Hi, Wi]
                xi = op.input[0]
                inputTensor = self.find_tensor_by_name(xi)
                [ni,ci,hi,wi] = input_shape = self.get_shape_by_name(xi)

                # 2\get [No, Co, Ho, Wo] from output
                ## output: [No, Co, Ho, Wo]
                [n,c,h,w] = output_shape = op.output_shape[0].dims[:]

                # 3\get [Nk, Ck, Hk, Wk] from  kernel weight
                ## weight: [in_channels, out_channels//groups, kernel_size[0], kernel_size[1]]
                ## The DI of SGS = in_channels,  The DO of SGS = out_channels,
                ## The DO of SGS cannot be represented by the C-dimensional data of weight, because C-dimensional data of weight equals to The DO of SGS//groups
                filter_name = op.input[1]
                filter_tensor = self.find_tensor_by_name(filter_name)
                [nk,ck,hk,wk] = filter_tensor.dims[:]
                di = nk
                do = ck*group
                filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(nk,ck,hk,wk)
                # reverse kernel
                filter_data = filter_data[:, :, ::-1, ::-1]
                filter_tensor.float_data[:] = filter_data.flat

                # 4\Category discussion
                ## SGS will divide DECONV into 3 categories
                ## case1: When group is 1
                ## case2: When group is not 1, it must also satisfy 'group == c == do and di == 1'
                ## case3: Other cases where group is not 1
                depthwise = False
                groupConv = False
                if group == c == ck and nk == 1 and group != 1: # The implementation of caffe and onnx convert to MACE is slightly different, so the judgment conditions are opposite
                    depthwise = True
                elif group > 1:
                    groupConv = True

                #creat transpose for input tensor
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = op_name + '_transpose'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0,2,3,1]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                    mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([xi])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend([ni,hi,wi,ci])
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                def computeOutSidePadding(input_shape,output_shape,strideH,strideW,kernel_shape,output_paddingH,output_paddingW):
                    [ni,ci,hi,wi] = input_shape
                    [n,c,h,w] = output_shape
                    input_pading_shape_H = hi + (hi - 1)*(strideH - 1)
                    input_pading_shape_W = wi + (wi - 1)*(strideW - 1)
                    round_func = math.floor
                    pad_h = round_func((h - 1 - output_paddingH - input_pading_shape_H + kernel_shape[0])/2)
                    pad_w = round_func((w - 1 - output_paddingW - input_pading_shape_W + kernel_shape[1])/2)
                    dilation_output_h = input_pading_shape_H + 2 * pad_h
                    dilation_output_w = input_pading_shape_W + 2 * pad_w
                    return pad_h,pad_w,dilation_output_h,dilation_output_w

                #                #
                # create dilation #
                #                #

                pad_h,pad_w,dilation_output_h,dilation_output_w = computeOutSidePadding(input_shape,output_shape,strideH,strideW,filter_tensor.dims[2:],output_paddingH,output_paddingW)
                op_dilation = self._SGSModel.op.add()
                op_dilation.name = 'Dilation' + op_name
                op_dilation.type = 'CUSTOM'
                # 1.add outside pad tensor
                outside_pad_name = op_.name + '_dilation' + '_outside_pad'
                outside_pad_data = [0,0,pad_h,pad_h + output_paddingH,pad_w,pad_w+output_paddingW,0,0] #[[n],[h],[w],[c]]
                outside_pad_shape = [4,2]
                self.add_tensor(self._SGSModel ,outside_pad_name, outside_pad_shape,
                    mace_pb2.DT_INT32, outside_pad_data)
                # 2.add inside pad tensor
                inside_pad_name = op_.name + '_dilation' + '_inside_pad'
                inside_pad_data = [0,int(strideH-1),int(strideW-1),0]
                inside_pad_shape = [4]
                self.add_tensor(self._SGSModel ,inside_pad_name, inside_pad_shape,
                    mace_pb2.DT_INT32, inside_pad_data)
                op_dilation.input.extend([output_op_transpose])
                op_dilation.input.extend([outside_pad_name])
                op_dilation.input.extend([inside_pad_name])
                output_name_dilation = op_.name + '_dilation' + '_output'
                op_dilation.output.extend([output_name_dilation])
                op_dilation.output_shape.add()
                op_dilation.output_shape[0].dims.extend([ni,dilation_output_h+output_paddingH,dilation_output_w+output_paddingW,ci])
                self._maceOpArray = np.append(self._maceOpArray,op_dilation)

                if depthwise:
                    #                       #
                    # create depthwiseconv   #
                    #                       #
                    op_dep_conv = self._SGSModel.op.add()
                    op_dep_conv.name = op_name + '_depthwise_conv'
                    op_dep_conv.type = 'DEPTHWISE_CONV_2D'
                    #add padding_type
                    paddingType_arg = op_dep_conv.arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    #set padding_type
                    paddingType_arg.i = self.pooling_paddingType['CAFFE']
                    strides_arg = op_dep_conv.arg.add()
                    strides_arg.name = 'strides'
                    strides_arg.ints.extend([1,1])
                    padding_arg = op_dep_conv.arg.add()
                    padding_arg.name = 'padding_values'
                    padding_arg.ints.extend([0,0,0,0])
                    op_dep_conv.input.extend([output_name_dilation])
                    depth_filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(1,ck,hk,wk)
                    depth_filter_data = np.transpose(depth_filter_data,(0,2,3,1))
                    filter_tensor.float_data[:] = depth_filter_data.flat
                    filter_tensor.dims[:] = [1,hk,wk,ck]
                    op_dep_conv.input.extend([filter_name])
                    op_dep_conv.input.extend([biase_name])
                    op_dep_conv.output[:] = op_.output[:]
                    output_shape = op_.output_shape[0].dims
                    if len(output_shape) == 4:
                        trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                    else:
                        trans_output_shape = output_shape
                    op_dep_conv.output_shape.add()
                    op_dep_conv.output_shape[0].dims.extend(trans_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_dep_conv)
                    self.OpOutputInsertTranspose(op_dep_conv)
                elif groupConv:
                    #                 #
                    # creat groupconv #
                    #                 #
                    op_conv = self._SGSModel.op.add()
                    op_conv.name = 'GroupConv' + op_name
                    op_conv.type = 'CUSTOM'
                    #add padding_type
                    paddingType_arg = op_conv.arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    paddingType_arg.i = self.pooling_paddingType['CAFFE']
                    padding_arg = op_conv.arg.add()
                    padding_arg.name = 'padding_values'
                    padding_arg.ints.extend([0,0,0,0])
                    strides_arg = op_conv.arg.add()
                    strides_arg.name = 'strides'
                    strides_arg.ints.extend([1,1])
                    group_arg = op_conv.arg.add()
                    group_arg.name = MaceKeyword.mace_group_str
                    group_arg.i = group
                    # deal filter tensor
                    depth_filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(nk,ck,hk,wk)
                    datalist = []
                    for i in six.moves.range(group):
                        data_slice = depth_filter_data[(nk//group)*i:(nk//group)*(i+1),:,:,:]
                        data_transpose = data_slice.transpose(1,0,2,3)
                        datalist.append(data_transpose)
                    depth_filter_data = np.concatenate(datalist, axis=0)
                    depth_filter_data = np.transpose(depth_filter_data,(0,2,3,1))
                    filter_tensor.float_data[:] = depth_filter_data.flat
                    filter_tensor.dims[:] = [ck*group,hk,wk,nk//group]
                    op_conv.input.extend([output_name_dilation])
                    op_conv.input.extend([filter_name])
                    op_conv.input.extend([biase_name])
                    op_conv.output[:] = op.output[:]
                    op_conv.output_shape.add()
                    output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                    op_conv.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_conv)
                    self.OpOutputInsertTranspose(op_conv)
                else:
                    #                #
                    # create conv2d   #
                    #                #
                    op_conv = self._SGSModel.op.add()
                    op_conv.name = op_name + '_conv'
                    op_conv.type = 'CONV_2D'
                    strides_arg = op_conv.arg.add()
                    strides_arg.name = 'strides'
                    strides_arg.ints.extend([1,1])
                    padding_arg = op_conv.arg.add()
                    padding_arg.name = 'padding_values'
                    padding_arg.ints.extend([0,0,0,0])
                    #add padding_type
                    paddingType_arg = op_conv.arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    #set padding_type
                    paddingType_arg.i = self.pooling_paddingType['CAFFE']
                    op_conv.input.extend([output_name_dilation])
                    depth_filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(nk,ck,hk,wk)
                    depth_filter_data = np.transpose(depth_filter_data,(0,2,3,1))
                    filter_tensor.float_data[:] = depth_filter_data.flat
                    filter_tensor.dims[:] = [nk,hk,wk,ck]
                    op_conv.input.extend([filter_name])
                    op_conv.input.extend([biase_name])
                    op_conv.output[:] = op_.output[:]
                    output_shape = op_.output_shape[0].dims
                    if len(output_shape) == 4:
                        trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                    else:
                        trans_output_shape = output_shape
                    op_conv.output_shape.add()
                    op_conv.output_shape[0].dims.extend(trans_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_conv)
                    self.OpOutputInsertTranspose(op_conv)
                self.remove_op_with_name(op_name)

    def split_DepthwiseConv1d(self, op):
        op_name = op.name
        if "\'" in op_name:
            op_name = op_name[2:-2]
        x = op.input[0]
        [ni,ci,wi] = self.get_shape_by_name(x)
        [no,co,wo] = op.output_shape[0].dims
        w = self.find_tensor_by_name(op.input[1]).float_data

        # create  transpose op
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = 'SGS_' + op_name + '#transpose'
        op_transpose.type = 'TRANSPOSE'
        op_transpose.input.extend([x])
        # set transpose's output_shape
        transpose_output_tensor_name = op_transpose.name + '_output_shape'
        transpose_output_tensor_data = [0, 2, 1]  # n,w,c
        transpose_output_tensor_shape = [3]
        self.add_tensor(self._SGSModel, transpose_output_tensor_name, transpose_output_tensor_shape,
                        mace_pb2.DT_INT32, transpose_output_tensor_data)
        op_transpose.input.extend([transpose_output_tensor_name])
        output_op_transpose = op_transpose.name + '_output'
        op_transpose.output.extend([output_op_transpose])
        op_transpose.output_shape.add()
        op_transpose.output_shape[0].dims.extend([ni,wi,ci])
        self._maceOpArray = np.append(self._maceOpArray, op_transpose)

        # creat reshape op
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '#reshape'
        op_reshape.type = 'RESHAPE'
        op_reshape.input.extend([output_op_transpose])
        # set reshape's output_shape
        reshape_output_tensor_name = op_reshape.name + '_output_shape'
        reshape_output_tensor_data = [ni, 1, wi, ci]  # n,h,w,c
        reshape_output_tensor_shape = [4]
        self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
        op_reshape.input.extend([reshape_output_tensor_name])
        output_op_reshape = op_reshape.name + '_output'
        op_reshape.output.extend([output_op_reshape])
        op_reshape.output_shape.add()
        op_reshape.output_shape[0].dims.extend([ni, 1, wi, ci])
        self._maceOpArray = np.append(self._maceOpArray, op_reshape)

        # create conv op
        weight = op.input[1]
        bias = op.input[2]
        arg = op.arg
        strideH,strideW = 0,0
        dilationH,dilationW = 1,1
        group = 1
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == MaceKeyword.mace_padding_values_str:
            #padding value should be divided by 2
                paddingL,paddingR,paddingT,paddingB = arg[i].ints
            elif name == MaceKeyword.mace_strides_str:
                strideH,strideW = arg[i].ints
            elif name == MaceKeyword.mace_dilations_str:
                dilationH,dilationW = arg[i].ints
            elif name == MaceKeyword.mace_kernel_str:
                kernelH,kernelW = arg[i].ints
            elif name == "group":
                group = arg[i].i

        # transpose 4dim weight
        weight_shape = self.get_shape_by_name(weight)
        weight_tensor = self.find_tensor_by_name(weight)
        if self.is_const_tensor(weight) == True:
            self.handle_const(op, "DECONV_2D", weight, weight_tensor, weight_shape)

        op_conv = self._SGSModel.op.add()
        op_conv.name = op_name + '_conv#1'
        op_conv.type = 'DEPTHWISE_CONV_2D'
        strides_arg = op_conv.arg.add()
        strides_arg.name = 'strides'
        strides_arg.ints.extend([strideH, strideW])
        dilation_arg = op_conv.arg.add()
        dilation_arg.name = 'dilations'
        dilation_arg.ints.extend([dilationH,dilationW])
        padding_arg = op_conv.arg.add()
        padding_arg.name = 'padding_values'
        padding_arg.ints.extend([paddingL, paddingR, paddingT, paddingB])
        #add padding_type
        paddingType_arg = op_conv.arg.add()
        paddingType_arg.name = MaceKeyword.mace_padding_str
        #set padding_type
        paddingType_arg.i = self.pooling_paddingType['CAFFE']
        op_conv.input.extend([output_op_reshape])
        op_conv.input.extend([op.input[1]])
        op_conv.input.extend([bias])
        output_op_conv = op_conv.name + '_output'
        op_conv.output.extend([output_op_conv])
        op_conv.output_shape.add()
        op_conv.output_shape[0].dims.extend([no, 1, wo, co])
        self._maceOpArray = np.append(self._maceOpArray, op_conv)

        # creat reshape op
        op_reshape2 = self._SGSModel.op.add()
        op_reshape2.name = op_name + '_reshape2'
        op_reshape2.type = 'RESHAPE'
        # op_reshape2.input.extend([output_op_conv])
        # set reshape's output_shape
        reshape2_output_tensor_name = op_reshape2.name + '_output_shape'
        reshape2_output_tensor_data = [no, wo, co]
        reshape2_output_tensor_shape = [3]
        self.add_tensor(self._SGSModel, reshape2_output_tensor_name, reshape2_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape2_output_tensor_data)
        op_reshape2.input.extend([output_op_conv])
        op_reshape2.input.extend([reshape2_output_tensor_name])
        # op_reshape2.output[:] = op.output[:]
        # op_reshape2.output_shape.extend(op.output_shape)
        output_op_reshape2 = op_reshape2.name + '_output'
        op_reshape2.output.extend([output_op_reshape2])
        op_reshape2.output_shape.add()
        op_reshape2.output_shape[0].dims.extend([no, wo, co])  # nchw
        self._maceOpArray = np.append(self._maceOpArray, op_reshape2)
        # self.remove_op_with_name(op_name)

        # create  transpose op
        op_transpose2 = self._SGSModel.op.add()
        op_transpose2.name = op_name + '#transpose2'
        op_transpose2.type = 'TRANSPOSE'
        op_transpose2.input.extend([output_op_reshape2])
        # set transpose's output_shape
        transpose2_output_tensor_name = op_transpose2.name + '_output_shape'
        transpose2_output_tensor_data = [0, 2, 1]  # n,c,w,h
        transpose2_output_tensor_shape = [3]
        self.add_tensor(self._SGSModel, transpose2_output_tensor_name, transpose2_output_tensor_shape,
                        mace_pb2.DT_INT32, transpose2_output_tensor_data)
        op_transpose2.input.extend([transpose2_output_tensor_name])
        op_transpose2.output[:] = op.output[:]
        #output_op_transpose2 = op_transpose2.name + '_output'
        #op_transpose2.output.extend([output_op_transpose2])
        op_transpose2.output_shape.add()
        op_transpose2.output_shape[0].dims.extend([no, co, wo])
        self._maceOpArray = np.append(self._maceOpArray, op_transpose2)

        #self.remove_op_with_name(op_name)

    def split_DepthwiseConv2d(self, op):
        # Transposed
      for op_ in self._SGSModel.op:
          if op_ == op:
            self.OpInputInsertTranspose(op_)
            op_name = op_.name
            input_data_name = op_.input[0]
            filter_name = op_.input[1]
            biase_name = op_.input[2]
            filter_tensor = self.find_tensor_by_name(op_.input[1])
            [n,c,h,w] = filter_tensor.dims[:]
            arg = op_.arg
            strideH,strideW = 0,0
            dilationH,dilationW = 1,1
            group = 1
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_padding_values_str:
                #padding value should be divided by 2
                    paddingL,paddingR,paddingT,paddingB = arg[i].ints
                elif name == MaceKeyword.mace_strides_str:
                    strideH,strideW = arg[i].ints
                elif name == MaceKeyword.mace_dilations_str:
                    dilationH,dilationW = arg[i].ints
                elif name == MaceKeyword.mace_kernel_str:
                    kernelH,kernelW = arg[i].ints
                elif name == "group":
                    group = arg[i].i
            #check pad value
            [ni,hi,wi,ci] = self.get_shape_by_name(input_data_name)
            [no,co,ho,wo] = op_.output_shape[0].dims
            kernel_dilationH =  (kernelH - 1) * dilationH + 1
            kernel_dilationW =  (kernelW - 1) * dilationW + 1
            ibH = (ho - 1) * strideH + kernel_dilationH
            ibW = (wo - 1) * strideW + kernel_dilationW
            need_check = False
            if ibW < wi + paddingL + paddingR:
                need_check = True
                paddingR = ibW - (wi + paddingL)
                if paddingR < 0:
                    paddingR = 0
            if ibH < hi + paddingT + paddingB:
                need_check = True
                paddingB = ibH - (hi + paddingT)
                if paddingB < 0:
                    paddingB = 0
            if need_check:
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                    #padding value should be divided by 2
                        arg[i].ints[:] = paddingL,paddingR,paddingT,paddingB

            if filter_tensor.float_data == []:
                #weight tensor is virable tensor
                #creat transpose
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '_transpose'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [3,1,2,0]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                    mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([filter_name])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend([w,c,h,n])
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                # replace op inputs
                op_.input[1] = output_op_transpose

            op_.type = "DEPTHWISE_CONV_2D"
            #add padding_type
            paddingType_arg = op_.arg.add()
            paddingType_arg.name = MaceKeyword.mace_padding_str
            #set padding_type
            paddingType_arg.i = self.pooling_paddingType['CAFFE']
            self._maceOpArray = np.append(self._maceOpArray,op_)
            self.OpOutputInsertTranspose(op_)

    def split_DepthToSpace(self, op):
        # unTransposed
        # SGS only support 4dims input case
        for op_ in self._SGSModel.op:
            if op_ == op:
                xi = op_.input[0]
                arg = op_.arg
                op_name = op_.name
                mode_type = 0
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'block_size':
                        block_size = arg[i].i
                    if name == 'mode':
                        mode_type = arg[i].i
                mace_check(len(self.get_shape_by_name(xi)) == 4, "SGS only support 4dims input case for onnx DepthToSpace!")
                [n,c,h,w] = self.get_shape_by_name(xi)
                c1 = int(c/(block_size*block_size))
                w1 = w//block_size
                h1 = h//block_size

                #add reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([xi])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                if mode_type == 0: #DCR mode
                    reshape_output_tensor_data = [n,block_size,block_size,c1,h,w]
                else: #CRD case
                    reshape_output_tensor_data = [n,c1,block_size,block_size,h,w]
                reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                if mode_type == 0:
                    op_reshape.output_shape[0].dims.extend([n,block_size,block_size,c1,h,w])
                else:
                    op_reshape.output_shape[0].dims.extend([n,c1,block_size,block_size,h,w])
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '_transpose#1'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                if mode_type == 0: #DCR mode
                    shape_tensor_data = [0, 3, 4, 1, 5, 2]
                else:
                    shape_tensor_data = [0, 1, 4, 2, 5, 3]
                shape_tensor_shape = [6]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                    mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([output_op_reshape])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend([n,c1,h,block_size,w,block_size])
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape#1'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_transpose])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [n, c1, h1, w1]
                reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape.output[:] =  op_.output[:]
                op_reshape.output_shape.extend(op_.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                self.remove_op_with_name(op_name)

    def split_Expand(self, op):
        # unTransposed
        def split_SameDimExpand(op):
            #[n,c,h,w] = op.output_shape[0].dims[:] #[n,c,h,w]
            #[ni,ci,hi,wi] = self.get_shape_by_name(op.input[0]) #
            input_shape = self.get_shape_by_name(op.input[0])
            output_shape = op.output_shape[0].dims[:]
            op_name = op.name
            xi = op.input[0]#NHWC in interpreter
            inputTensor = self.find_tensor_by_name(op.input[0])
            scale = []
            #mace_check(len(input_shape) == len(output_shape), "Expand only support in/out tensors with same dims number")
            for i in six.moves.range(len(input_shape)):
                scale.extend([output_shape[i]//input_shape[i]])
            #creat tiles
            op_tile = self._SGSModel.op.add()
            op_tile.name = op_name + '_tile'
            op_tile.type = 'TILE'
            multiples_tensor_name = op_tile.name + '_multiples'
            multiples_tensor_data = scale
            multiples_tensor_shape = [len(input_shape)]
            self.add_tensor(self._SGSModel ,multiples_tensor_name, multiples_tensor_shape,
                mace_pb2.DT_INT32, multiples_tensor_data)
            op_tile.input.extend([xi])
            op_tile.input.extend([multiples_tensor_name])
            op_tile.output[:] = op.output[:]
            op_tile.output_shape.extend(op.output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_tile)
            #remove op
            self.remove_op_with_name(op_name)


        def split_UnequalDimExpand(op):
            input_shape = self.get_shape_by_name(op.input[0])
            inputTensor = self.find_tensor_by_name(op.input[0])
            input_shape_num = 1
            output_shape_num = 1
            for i in range(len(input_shape)):
                input_shape_num = input_shape_num * input_shape[i]
            output_shape = op.output_shape[0].dims[:]
            for i in range(len(output_shape)):
                output_shape_num = output_shape_num * output_shape[i]
            output_dim = len(output_shape)
            op_name = op.name
            if input_shape_num != output_shape_num:
                # create reshape
                op_output_shape = []
                for i in six.moves.range(output_dim):
                    op_output_shape.append(1)
                for i in six.moves.range(len(input_shape)):
                    op_output_shape[-(i+1)] = input_shape[-(i+1)]
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = op_output_shape
                reshape_output_tensor_shape = [len(op_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([op.input[0]])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_output'
                op_reshape.output.extend([op_reshape_output_name])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(op_output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                # create tile
                op_tile = self._SGSModel.op.add()
                op_tile.name = op_name + '_tile'
                op_tile.type = 'TILE'
                multiples_tensor_name = op_tile.name + '_multiples'
                scale = []
                for i in six.moves.range(len(output_shape)):
                    scale.extend([output_shape[i]//op_output_shape[i]])
                multiples_tensor_data = scale
                multiples_tensor_shape = [len(output_shape)]
                self.add_tensor(self._SGSModel ,multiples_tensor_name, multiples_tensor_shape,
                    mace_pb2.DT_INT32, multiples_tensor_data)
                op_tile.input.extend([op_reshape_output_name])
                op_tile.input.extend([multiples_tensor_name])
                op_tile.output[:] = op.output[:]
                op_tile.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_tile)
                #remove op
                self.remove_op_with_name(op_name)
            else:
                # create reshape
                op_output_shape = []
                for i in six.moves.range(output_dim):
                    op_output_shape.append(1)
                for i in six.moves.range(len(input_shape)):
                    op_output_shape[-(i+1)] = input_shape[-(i+1)]
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = op_output_shape
                reshape_output_tensor_shape = [len(op_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([op.input[0]])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape.output[:] = op.output[:]
                op_reshape.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                #remove op
                self.remove_op_with_name(op_name)

        # main
        for op_ in self._SGSModel.op:
            if op_ == op:
                xi = op_.input[0]
                yi = op_.output[0]
                expand_shape = self.get_shape_by_name(yi)
                inputShape = self.get_shape_by_name(xi)
                if len(inputShape) == len(expand_shape):
                    split_SameDimExpand(op_)
                elif len(inputShape) != len(expand_shape):
                    split_UnequalDimExpand(op_)

    def split_Eltwise(self, op):
        # Transposed
        def doReshape(op):
            op_ = op
            for index,inputName in enumerate(op_.input):
                if self.is_const_tensor(inputName) == True:
                    tensor = self.find_tensor_by_name(inputName)
                    input_shape = copy.deepcopy(self.get_shape_by_name(inputName))
                    if len(input_shape) != 4:
                        self.handle_const(op_, "ELTWISE", inputName, tensor, input_shape, index)
                elif self.is_const_tensor(inputName) == False:
                    tensor = self.find_tensor_by_name(inputName)
                    input_shape = self.get_shape_by_name(inputName)
                    if len(input_shape) != 4:
                        if (len(input_shape) == 3 and input_shape[-2:] == [1,1]):
                            # create reshape, Need to multiply vector [3,1,1] into [3]
                            op_reshape = self._SGSModel.op.add()
                            op_reshape.name = op_name + '#SGS_RESHAPE'
                            op_reshape.type = 'RESHAPE'
                            op_reshape.input.extend([inputName])
                            # set reshape's output_shape
                            reshape_output_tensor_name = op_reshape.name + '_output_shape'
                            reshape_output_tensor_data = [input_shape[0]]  # [c,1,1] to [c]
                            reshape_output_tensor_shape = [1]
                            self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                            mace_pb2.DT_INT32, reshape_output_tensor_data)
                            op_reshape.input.extend([reshape_output_tensor_name])
                            output_op_reshape = op_reshape.name + '_output'
                            op_reshape.output.extend([output_op_reshape])
                            op_reshape.output_shape.add()
                            op_reshape.output_shape[0].dims.extend([input_shape[0]])  # nchw
                            self._maceOpArray = np.append(self._maceOpArray, op_reshape)
                            self.remove_tensor_by_name(inputName)
                            op_.input[index] = output_op_reshape
                        else:
                            mace_check(False,"if inputs are both variable tensor, not support yet!")

        # main
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_element_type_str:
                        type = arg[i].i
                        # 0:SUM 1:SUB 2:PROD 3:DIV 4:MIN 5:MAX 6:NEG 7:ABS 8:SQR_DIFF 9:POW 10:EQUAL 11:FLOOR_DIV
                        if type == 0:
                            op_.type = 'ADD'
                        elif type == 1:
                            op_.type = 'SUB'
                        elif type == 2:
                            op_.type = 'MUL'
                        elif type == 4:
                            op_.type = 'MINIMUM'
                        elif type == 5:
                            op_.type = 'MAXIMUM'
                        elif type == 6:
                            op_.type = 'NEG'
                        elif type == 3:
                            op_.type = 'DIV'
                        elif type == 7:
                            op_.type = 'ABS'
                        elif type == 9:
                            op_.type = 'POW'
                            return self.split_Pow(op_)
                        elif type == 10:
                            op_.type = 'EQUAL'
                        else:
                            mace_check(False, "Does not support this eltwise op type %s" % op_name)
                if len(op_.input) > 1 :
                    # convert const tensors' shape:
                    #       if one of input dim = 4,
                    #       another input dim < 4,
                    #       < 4 need change to 4 dim.

                    input_shape1 = self.get_shape_by_name(op_.input[0])
                    input_shape2 = self.get_shape_by_name(op_.input[1])
                    needReshape = False
                    if len(input_shape1) == 4 or len(input_shape2) == 4:
                        needReshape = True
                        doReshape(op_)
                    inputTensor1 = self.find_tensor_by_name(op_.input[0])
                    inputTensor2 = self.find_tensor_by_name(op_.input[1])
                    if op_.type in ['ADD', 'SUB', 'MUL', 'DIV']:
                        self.cast_tensor(inputTensor1, mace_pb2.DT_FLOAT)
                        self.cast_tensor(inputTensor2, mace_pb2.DT_FLOAT)
                    if len(inputTensor1.dims) == 4 or len(inputTensor2.dims) == 4:
                        self.OpInputInsertTranspose(op_)
                        self._maceOpArray = np.append(self._maceOpArray, op_)
                        self.OpOutputInsertTranspose(op_)
                    else:
                        self._maceOpArray = np.append(self._maceOpArray, op_)
                else:
                    if op_.type == 'MINIMUM' or op_.type == 'MAXIMUM':
                        # create second input to store min/max value
                        default_min = np.finfo(np.float32).min
                        default_max = np.finfo(np.float32).max
                        for i in six.moves.range(len(arg)):
                            name = arg[i].name
                            if name == MaceKeyword.mace_coeff_str:
                                value = arg[i].floats
                                for j in six.moves.range(len(value)):
                                    if value[j] == default_min or value[j] == default_max:
                                        continue
                                    value_shape = [1]
                                    data = [value[j]]
                                    value_tensor_name = op.name + '_value'
                                    self.add_tensor(self._SGSModel, value_tensor_name, value_shape,
                                                    mace_pb2.DT_FLOAT, data)
                                    op_.input.extend([value_tensor_name])
                        self._maceOpArray = np.append(self._maceOpArray, op_)
                    if op_.type == 'NEG':
                        self.OpInputInsertTranspose(op_)
                        self._maceOpArray = np.append(self._maceOpArray,op_)
                        self.OpOutputInsertTranspose(op_)
                    else:
                        self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Pow(self, op):
        # unTransposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op.name
                base = op.input[0]
                inputTensor1 = self.find_tensor_by_name(base)
                power_data = []
                power = 0
                if len(op.input) == 2:
                    const_name = op.input[1]
                    if self.is_const_tensor(const_name) == True:
                        const_tensor = self.find_tensor_by_name(const_name)
                        if const_tensor.data_type == mace_pb2.DT_INT32:
                            for i in six.moves.range(len(const_tensor.int32_data)):
                                power_data.extend([float(const_tensor.int32_data[i])])
                        elif const_tensor.data_type == mace_pb2.DT_FLOAT:
                            for i in six.moves.range(len(const_tensor.float_data)):
                                power_data.extend([float(const_tensor.float_data[i])])
                        else:
                            mace_check(False, "Does not support param's data type %s" % const_tensor.data_type)

                        temp = set(power_data)
                        mace_check(len(temp) == 1, "POW only support exponent tensor with all same data")
                        power =  power_data[0]
                        mace_check(power != 0 and power != 1, "POW does not support exponent as 0 and 1")
                    else:
                        mace_check(False, "Does not support unconst tensor %s" % const_tensor)
                elif len(op.input) == 1:
                    power = 0.5 #sqrt
                else:
                    mace_check(False, "wrong input number for Pow or Sqrt")

                # create pow
                if power == 0.5:
                    self.OpInputInsertTranspose(op_)
                    op_.type = 'SQRT'
                    # sqrt only needs input0
                    if len(op.input) == 2:
                        op_.input.pop()
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)

                elif power % 1 != 0:
                    #mace_check(False, "POW only support non-integer exponent 0.5")
                    self.OpInputInsertTranspose(op_)
                    op_.name = 'CustomPow' + op_name
                    op_.type = 'CUSTOM'
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)

                else:
                    #loop add mul
                    mace_check(power > 1, "POW only support integer larger than 1")
                    loop_num = int(power)-1
                    for i in six.moves.range(loop_num):
                        op_mul = self._SGSModel.op.add()
                        op_mul.name = op_name + '_MUL#' + str(i)
                        op_mul.type = 'MUL'
                        op_mul.input.extend([base])
                        if i == 0:
                            op_mul.input.extend([base])
                        else:
                            op_mul.input.extend([output_name_mul])
                        if i == (loop_num - 1):
                            op_mul.output[:] = op.output[:]
                        else:
                            output_name_mul = op_mul.name + '_output#' + str(i)
                            op_mul.output.extend([output_name_mul])
                        op_mul.output_shape.extend(op.output_shape)
                        #tensor data is variable,so didn't creat new tensor
                        self._maceOpArray = np.append(self._maceOpArray,op_mul)
                    # remove original op
                    self.remove_op_with_name(op_name)

    def split_Exp(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                    self.OpInputInsertTranspose(op_)
                    op_.type = "EXP"
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)

    def split_Where(self, op):
        def is_need_broadcast(len_dims, tensor_dims, output_dims):
            tmp_list = []
            tile_param = []
            if len_dims == 4:
                if (tensor_dims == [1, 1, 1, 1] or tensor_dims == [1, 1, 1, output_dims[1]] # nhwc needn't broadcast
                        or tensor_dims == output_dims):
                    return False, None
                else:
                    [n,h,w,c] = tensor_dims
                    [N,C,H,W] = output_dims
                    if n != N:
                        tile_param.append(N)
                    else:
                        tile_param.append(1)
                    if h != H:
                        tile_param.append(H)
                    else:
                        tile_param.append(1)
                    if w != W:
                        tile_param.append(W)
                    else:
                        tile_param.append(1)
                    if c != C:
                        tile_param.append(C)
                    else:
                        tile_param.append(1)

                    # for index, i in enumerate(output_shape):
                    #     if i != wher.e_const_tensor.dims[index]:
                    #         tile_param.append(output_shape[index])
                    #     else:
                    #         tile_param.append(1)
                    return True, tile_param
            elif len_dims == 0 or len_dims == 1:
                return False, None
            else:
                for x in range(len(tensor_dims)-1):
                    tmp_list.append(1)
                tmp_list.append(output_dims[-1])
                if tensor_dims == tmp_list or tensor_dims == output_dims:
                    return False, None
                else:
                    for index, i in enumerate(output_shape):
                        if i != where_const_tensor.dims[index]:
                            tile_param.append(output_shape[index])
                        else:
                            tile_param.append(1)
                    return True, tile_param


        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_name = op_.name
                output = op_.output[0]
                output_shape = self.get_shape_by_name(output)
                for index1, m in enumerate(op_.input):
                    input_tensor = self.find_tensor_by_name(m)
                    has_const = False
                    if input_tensor.data_type == mace_pb2.DT_FLOAT:
                        if len(input_tensor.float_data) != 0:
                            where_const_tensor = input_tensor
                            where_const_data = where_const_tensor.float_data
                            for idx, data in enumerate(where_const_data):
                                if data == np.inf or data == -np.inf:
                                    print(f'Got the data inf of tensor({where_const_tensor.name}), think that the data is invalid, '
                                        'convert it to 0, so that it can be supported')
                                    where_const_data[idx] = 0
                            has_const = True
                    elif input_tensor.data_type == mace_pb2.DT_INT32:
                        if len(input_tensor.int32_data) != 0:
                            where_const_tensor = input_tensor
                            where_const_data = where_const_tensor.int32_data
                            has_const = True
                    else:
                        mace_check(False, "Does not support param's data type %s" % input_tensor.data_type)

                    if has_const:
                        where_name_index = index1
                        is_need, tile_param = is_need_broadcast(len(where_const_tensor.dims),
                                                    where_const_tensor.dims, output_shape)
                        if is_need:
                            where_const_ndarray = np.array(where_const_data)
                            where_const_ndarray = np.reshape(where_const_ndarray, where_const_tensor.dims)
                            where_const_tensor_1 = np.tile(where_const_ndarray, tile_param)
                            where_const_tensor_1 = where_const_tensor_1.flatten()
                            where_const_tensor_1 = list(where_const_tensor_1)
                            op_where_name = op_name + '_where#' + str(index1)
                            if len(output_shape) == 4:
                                output_shape_change = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                            else:
                                output_shape_change = output_shape
                            self.add_tensor(self._SGSModel, op_where_name, output_shape_change,
                                            mace_pb2.DT_FLOAT, where_const_tensor_1)
                            op_.input[where_name_index] = op_where_name

                op_.type = "SELECT"
                self._maceOpArray = np.append(self._maceOpArray, op_)
                self.OpOutputInsertTranspose(op_)

    def split_Greater(self, op):
        def doReshape(op):
            op_ = op
            for index,inputName in enumerate(op_.input):
                if self.is_const_tensor(inputName) == True:
                    tensor = self.find_tensor_by_name(inputName)
                    input_shape = copy.deepcopy(self.get_shape_by_name(inputName))
                    if len(input_shape) != 4:
                        self.handle_const(op_, "ELTWISE", inputName, tensor, input_shape, index)
                elif self.is_const_tensor(inputName) == False:
                    tensor = self.find_tensor_by_name(inputName)
                    input_shape = self.get_shape_by_name(inputName)
                    if len(input_shape) != 4:
                        if (len(input_shape) == 3 and input_shape[-2:] == [1,1]):
                            # create reshape, Need to multiply vector [3,1,1] into [3]
                            op_reshape = self._SGSModel.op.add()
                            op_reshape.name = op_name + '#SGS_RESHAPE'
                            op_reshape.type = 'RESHAPE'
                            op_reshape.input.extend([inputName])
                            # set reshape's output_shape
                            reshape_output_tensor_name = op_reshape.name + '_output_shape'
                            reshape_output_tensor_data = [input_shape[0]]  # [c,1,1] to [c]
                            reshape_output_tensor_shape = [1]
                            self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                            mace_pb2.DT_INT32, reshape_output_tensor_data)
                            op_reshape.input.extend([reshape_output_tensor_name])
                            output_op_reshape = op_reshape.name + '_output'
                            op_reshape.output.extend([output_op_reshape])
                            op_reshape.output_shape.add()
                            op_reshape.output_shape[0].dims.extend([input_shape[0]])  # nchw
                            self._maceOpArray = np.append(self._maceOpArray, op_reshape)
                            self.remove_tensor_by_name(inputName)
                            op_.input[index] = output_op_reshape
                        else:
                            mace_check(False,"if inputs are both variable tensor, not support yet!")

        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "GREATER"
                if len(op_.input) > 1 :
                    # convert const tensors' shape:
                    #       if one of input dim = 4,
                    #       another input dim < 4,
                    #       < 4 need change to 4 dim.

                    input_shape1 = self.get_shape_by_name(op_.input[0])
                    input_shape2 = self.get_shape_by_name(op_.input[1])
                    needReshape = False
                    if len(input_shape1) == 4 or len(input_shape2) == 4:
                        needReshape = True
                        doReshape(op_)
                    inputTensor1 = self.find_tensor_by_name(op_.input[0])
                    inputTensor2 = self.find_tensor_by_name(op_.input[1])
                    if len(inputTensor1.dims) == 4 or len(inputTensor2.dims) == 4:
                        self.OpInputInsertTranspose(op_)
                        self._maceOpArray = np.append(self._maceOpArray, op_)
                        self.OpOutputInsertTranspose(op_)
                    else:
                        self._maceOpArray = np.append(self._maceOpArray, op_)
                else:
                    self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Elu(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "ELU"
                op_.name = 'ELU'+ op.name
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_scatternd(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = 'CUSTOM'
                op_.name = 'Customized_ScatterND' + op.name
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Erf(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "CUSTOM"
                op_.name = 'Erf' + op.name
                self._maceOpArray = np.append(self._maceOpArray, op_)
                self.OpOutputInsertTranspose(op_)

    def create_Transpose_op(self, op_, opName, inputTensorName, permutation, inputTensorShape = None,need_create_out = True):
        self._tensorNumCreatedBySgs +=1
        op_name = opName
        input = inputTensorName
        if inputTensorShape == None:
            input_shape = self.get_shape_by_name(input)
        else:
            input_shape = inputTensorShape
        if self.is_const_tensor(input):
            input_tensor = self.find_tensor_by_name(input)
            if self.find_tensor_by_name(input).int32_data != []:
                const_input_ori_data = self.find_tensor_by_name(input).int32_data
            else:
                const_input_ori_data = self.find_tensor_by_name(input).float_data
            if len(const_input_ori_data) != 0:
                data = np.array(const_input_ori_data)
                data = data.reshape(input_shape)
                data = data.transpose(permutation)
                data = list(data.flat)
                new_data = copy.deepcopy(data)
                tmp_dim = permutation
                tmp_shape = [i for i in six.moves.range(len(input_shape))]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = input_shape[tmp_dim[i]]
                for i in six.moves.range(len(tmp_dim)):
                    input_tensor.dims[i] = tmp_shape[i]
                const_tensor_name = op_name + '_const'
                const_tensor_shape = tmp_shape
                const_tensor_data = data
                self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                mace_pb2.DT_FLOAT, const_tensor_data)
                new_input = const_tensor_name
                del input_tensor
        else:
            tmp_dim = permutation
            tmp_shape = [i for i in six.moves.range(len(input_shape))]
            for i in six.moves.range(len(tmp_dim)):
                tmp_shape[i] = input_shape[tmp_dim[i]]
            op_transpose = self._SGSModel.op.add()
            op_transpose.name = op_name + '_transpose' + str(self._tensorNumCreatedBySgs)
            op_transpose.type = 'TRANSPOSE'
            shape_tensor_name = op_transpose.name + '_shape'
            shape_tensor_data = permutation
            shape_tensor_shape = [len(permutation)]
            self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape,
                mace_pb2.DT_INT32, shape_tensor_data)
            op_transpose.input.extend([input])
            op_transpose.input.extend([shape_tensor_name])
            if need_create_out:
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(tmp_shape)
                new_input = output_op_transpose
            else:
                op_transpose.output[:] = op_.output[:]
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(tmp_shape)
                new_input = op_.output[:]
            self._maceOpArray = np.append(self._maceOpArray,op_transpose)

        return new_input

    def create_Reshape_op(self, op_, opName, inputTensorName, outputTensorShape, need_create_out = True):
        self._tensorNumCreatedBySgs +=1
        op_name = opName
        input = inputTensorName
        shape = outputTensorShape
        if self.is_const_tensor(input):
            input_shape = self.get_shape_by_name(input)
            input_tensor = self.find_tensor_by_name(input)
            if self.find_tensor_by_name(input).int32_data != []:
                const_input_ori_data = self.find_tensor_by_name(input).int32_data
            else:
                const_input_ori_data = self.find_tensor_by_name(input).float_data
            if len(const_input_ori_data) != 0:
                data = np.array(const_input_ori_data)
                data = data.reshape(input_shape)
                data = data.reshape(shape)
                data = list(data.flat)
                new_data = copy.deepcopy(data)
                const_tensor_name = op_name + '_const'
                const_tensor_shape = shape
                const_tensor_data = data
                self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                mace_pb2.DT_FLOAT, const_tensor_data)
                new_input = const_tensor_name
                del input_tensor
        else:
            # create reshape
            op_reshape = self._SGSModel.op.add()
            op_reshape.name = op_name + '_reshape' + str(self._tensorNumCreatedBySgs)
            op_reshape.type = 'RESHAPE'
            op_reshape.input.extend([input])
            reshape_output_tensor_name = op_reshape.name + '_output_shape'
            reshape_output_tensor_data = shape#n,h,w,c
            reshape_output_tensor_shape = [len(shape)]
            self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                            mace_pb2.DT_INT32, reshape_output_tensor_data)
            op_reshape.input.extend([reshape_output_tensor_name])
            if need_create_out:
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(shape)#n,c,h,w
                new_input = output_op_reshape
            else:
                op_reshape.output[:] = op_.output[:]
                output_shape = op_.output_shape
                op_reshape.output_shape.extend(output_shape)
                new_input = op_.output[:]
            self._maceOpArray = np.append(self._maceOpArray,op_reshape)

        return new_input

    def create_Eltwise_op(self, op_, opName, opType, inputTensor1Name, inputTensor2Name, outputTensorName=None, outputTensorShape=None, need_create_out = True):
        x1 = inputTensor1Name
        x2 = inputTensor2Name
        op_type = opType
        type_str = op_type.lower()
        op_name = opName
        op_elt = self._SGSModel.op.add()
        op_elt.name = op_name + type_str
        op_elt.type = op_type
        op_elt.input.extend([x1])
        op_elt.input.extend([x2])
        if need_create_out:
            mace_check(outputTensorName!=None,"False! must give a new out tensor name!")
            op_elt.output.extend([outputTensorName])
            op_elt.output_shape.add()
            op_elt.output_shape[0].dims.extend(outputTensorShape)
            self.add_tensor(self._SGSModel, outputTensorName, outputTensorShape, mace_pb2.DT_FLOAT,
                            None)
        else:
            op_elt.output[:] = op_.output[:]
            output_shape = op_.output_shape
            outputTensorName = op_.output[:]
            op_elt.output_shape.extend(output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_elt)

        return outputTensorName

    def create_Sum_op(self, op_, opName, inputTensorName, axisInfo, inputTensorShape, keepdimFlag = True, need_create_out = True):
        self._tensorNumCreatedBySgs +=1
        op_name = opName
        input = inputTensorName
        axis_data = [axisInfo]
        output_shape = []
        if keepdimFlag:
            keepdim = 1
            for a in six.moves.range(len(inputTensorShape)):
                if a not in axis_data:
                    output_shape.append(inputTensorShape[a])
                else:
                    output_shape.append(1)
            #output_shape = [inputTensorShape[a] if a not in axis_data else 1 for a in six.moves.range(len(inputTensorShape))]  # keepdim = True
        else:
            keepdim = 0
            for a in six.moves.range(len(inputTensorShape)):
                if a not in axis_data:
                    output_shape.append(inputTensorShape[a])
            #output_shape = [inputTensorShape[a] for a in six.moves.range(len(inputTensorShape)) if a not in axis_data] # keepdim = False

        # creat axis tensor
        axis_tensor_name = op_name + '_axis'
        axis_tensor_data = axis_data
        axis_tensor_shape = [len(axis_data)]
        self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
        mace_pb2.DT_INT32, axis_tensor_data)

        #creat reduceSum
        op_reduceSum = self._SGSModel.op.add()
        op_reduceSum.name = op_name + '_reduceSum' + str(self._tensorNumCreatedBySgs)
        op_reduceSum.type = 'SUM'
        keepdims_arg = op_reduceSum.arg.add()
        keepdims_arg.name = 'keepdims'
        keepdims_arg.i = keepdim
        op_reduceSum.input.extend([input])
        op_reduceSum.input.extend([axis_tensor_name])
        if need_create_out:
            output_op_reduceSum = op_reduceSum.name + '_output'
            op_reduceSum.output.extend([output_op_reduceSum])
            op_reduceSum.output_shape.add()
            op_reduceSum.output_shape[0].dims.extend(output_shape)
            output_name = output_op_reduceSum
        else:
            op_reduceSum.output[:] = op_.output[:]
            output_shape = op_.output_shape
            output_name = op_.output[:]
            op_reduceSum.output_shape.extend(output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)

        return output_name, output_shape

    def split_Einsum_custom(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                left = nkcvt = op_.input[0]
                left_shape = [n,k,c,t,v] = self.get_shape_by_name(left)
                right = kvw = op_.input[1]
                right_shape = [k,v,w] = self.get_shape_by_name(right)
                output_shape = [n,c,t,w] = self.get_shape_by_name(op_.output[0])

                # create transpose for left input
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '_transposeLeft'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shapeOut'
                shape_tensor_data = [2,3,1,4,0]
                shape_tensor_shape = [5]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([left])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_outputLeft'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                tmp_shape = [c,t,k,v,n]
                op_transpose_shape = copy.deepcopy(tmp_shape)
                op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                # create reshape for left input after transpose
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshapeleft'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_transpose])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [1,c*t,k*v]#n,h,w,c
                reshape_output_tensor_shape = [3]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend([1,c*t,k*v])#n,c,h,w
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                # create matmul and reshape/transpose right
                right_tensor = self.find_tensor_by_name(right)
                right_shape = [1,k*v,w]
                if self.is_const_tensor(right):
                    self.handle_const(op_, "MATMUL", right, right_tensor, right_shape)
                else:
                    new_right_after_reshape_name = self.create_Reshape_op(op_, op_name + '_kvw', right, right_shape)
                    transpose_dim = [0,2,1]
                    new_right_after_transpose_name = self.create_Transpose_op(op_, op_name + '_kvw', new_right_after_reshape_name, transpose_dim,inputTensorShape = right_shape)
                    op_.input[1] = new_right_after_transpose_name
                op_batchmatmul = self._SGSModel.op.add()
                op_batchmatmul.name = op_name + '_batchmatmul'
                op_batchmatmul.type = "BATCH_MATMUL"
                op_batchmatmul.input.extend([output_op_reshape])    # input
                op_batchmatmul.input.extend([op_.input[1]])         # weight
                output_op_batchmatmul = op_batchmatmul.name + '_outputLeft'
                op_batchmatmul.output.extend([output_op_batchmatmul])
                op_batchmatmul.output_shape.add()
                tmp_shape = [1,c*t,w]
                op_batchmatmul_shape = copy.deepcopy(tmp_shape)
                op_batchmatmul.output_shape[0].dims.extend(op_batchmatmul_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_batchmatmul)

                # create reshape for output
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape#1'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_batchmatmul])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [1,c,t,w]#n,h,w,c
                reshape_output_tensor_shape = [4]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape.output[:] = op.output[:]
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend([1,c,t,w])#n,c,h,w
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                self.remove_op_with_name(op_name)


    def split_Einsum(self, op):
        # equation:'aij,ajk->aik'
        #           out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
        '''
            left        right
              |           |
          reshape     (transpose)
              |           |
              |        reshape
              |-----------|
                    |
                   mul
                    |
               (reducesum)
                    |
               (transpose)
                    |
        '''

        def einsum_equation_preprocessor(op_, einsum_equation, left_equation_exist = False, right_equation_exist = False, out_equation_exist = False):
            # 'left_equation,right_equation->out_equation'
            split_equation_str = '->'
            split_input_str = ','
            ellipsis_str = '.'
            # 1\ split with '->' for equation
            if split_equation_str in einsum_equation:
                out_equation_exist = True
                [in_equation,out_equation] = einsum_equation.split('->')
                out_equation = list(out_equation)
            else:
                mace_check(False,'Not support yet!')
            # 2\ split with ','  for input
            if split_input_str in in_equation:
                left_equation_exist = True
                right_equation_exist = True
                [left_equation,right_equation] = in_equation.split(',')
                left_equation = list(left_equation)
                right_equation = list(right_equation)
            else:
                mace_check(False,'Not support yet!')
            # 3\ record all dimensions that appear in inputs and find summation dimensions
            in_equation = [left_equation,right_equation]

            return left_equation, right_equation, in_equation, out_equation

        def einsum_uniform_equation_index(op_, left_equation, right_equation, in_equation, out_equation):
            # create a-z,A-Z map table for search
            LetterToIndex = {}
            for i in six.moves.range(26):
                LetterToIndex[chr(ord('a')+i)] = i
            for i in six.moves.range(26):
                LetterToIndex[chr(ord('A')+i)] = 26 + i

            # create search record list,
            # 'count' record letter number of occurrences from all inputs,
            # 'index' record letter order of appearance from all inputs
            letter_to_count = [ 0 for i in six.moves.range(52)]
            letter_to_index = [-1 for i in six.moves.range(52)]
            subscript_indices_to_dim_value = []  # subscript 下标
            subscript_indices_to_last_input = []

            # get shape-subscript mapping of in/out
            num_subscript_indices = -1
            input_subscript_indices = []
            for input_index in six.moves.range(len(in_equation)):
                inputname = op_.input[input_index]
                dim_counter = 0
                current_equation_shape = self.get_shape_by_name(inputname)
                current_subscript_indices = []
                for j in six.moves.range(len(in_equation[input_index])):
                    current_equation = in_equation[input_index]
                    subscript_label = current_equation[j]
                    if subscript_label == '.':
                        pass
                    else:
                        letter_index = LetterToIndex[subscript_label] # b,h,l,k
                        dim_value = current_equation_shape[dim_counter] # 1,3,2,4
                        if (letter_index == -1):
                            mace_check(False,"The only subscript labels allowed are lower-cased letters (a-z) and upper-cased letters (A-Z)!")

                        elif letter_to_count[letter_index] == 0:
                            num_subscript_indices  += 1
                            letter_to_index[letter_index] = num_subscript_indices
                            subscript_indices_to_dim_value.append(dim_value)
                            subscript_indices_to_last_input.append(input_index)
                        else:
                            mapped_index = letter_to_index[letter_index]
                            subscript_indices_to_last_input[mapped_index] = input_index
                            if (subscript_indices_to_dim_value[mapped_index] != dim_value):
                                if (subscript_indices_to_dim_value[mapped_index] == 1):
                                    subscript_indices_to_dim_value[mapped_index] = dim_value
                                elif dim_value != 1:
                                    mace_check(False,"Please check input shapes/equation provided!")
                        current_subscript_indices.append(letter_to_index[letter_index])
                        dim_counter += 1
                        count = letter_to_count[letter_index]
                        letter_to_count[letter_index] = count + 1
                        if (dim_counter > len(in_equation[input_index])):
                            mace_check(False,'too many subscript labels!')
                input_subscript_indices.append(current_subscript_indices)
            # output
            subscript_indices_to_output_indices = [-1 for i in six.moves.range(len(letter_to_index)) if letter_to_index[i] != -1 ]
            output_letter_to_count = [ 0 for i in six.moves.range(52)]
            output_dims = []
            output_dim_counter = 0
            for i in six.moves.range(len(out_equation)):
                subscript_label = out_equation[i]
                if subscript_label == '.':
                    pass
                else:
                    letter_index = LetterToIndex[subscript_label] # b,h,l,k
                    if (letter_index == -1):
                        mace_check(False,"The only subscript labels allowed are lower-cased letters (a-z) and upper-cased letters (A-Z)!")
                    if (output_letter_to_count[letter_index] != 0):
                        mace_check(False,"Output subscript contains repeated letters!")
                    else:
                        output_letter_to_count[letter_index] = 1
                        mapped_index = letter_to_index[letter_index]
                        if (mapped_index == -1):
                            mace_check(False,"Output subscript contains letters not seen in the inputs!")
                        output_dims.append(subscript_indices_to_dim_value[mapped_index])
                        subscript_indices_to_last_input[mapped_index] = -1 # -1 means can't be reduced
                        subscript_indices_to_output_indices[mapped_index] = output_dim_counter
                        output_dim_counter += 1
            reduce_axes_subscript = [ k for k in six.moves.range(len(subscript_indices_to_last_input)) if subscript_indices_to_last_input[k] != -1]
            mul_outshape = subscript_indices_to_dim_value

            return input_subscript_indices,subscript_indices_to_output_indices, letter_to_index, mul_outshape, reduce_axes_subscript

        def PreprocessInputs(op_, in_equation, input_subscript_indices, letter_to_index, mul_outshape ):
            preprocessed_inputs_reshape_dims = []
            #  1) Making them all of the same rank                                                                                                                                                         │
            #  2) The axes order in all the inputs are to be made the same
            input_subscript_indices = input_subscript_indices # [0,1,2,3,4],[1,4,5]
            homogenized_input_dims = [1 for i in six.moves.range(len(letter_to_index)) if letter_to_index[i] != -1 ]
            for i in six.moves.range(len(input_subscript_indices)):
                subscript_indices_to_input_index = [-1 for j in six.moves.range(len(letter_to_index)) if letter_to_index[j] != -1 ]
                current_subscript_indices = input_subscript_indices[i]
                # homogenized_input_dims(num_subscript_indices, 1)
                dim_index_in_preprocessed_input = 0
                dim_index_in_original_input = 0
                for subscript_index in six.moves.range(len(current_subscript_indices)):
                    if subscript_indices_to_input_index[current_subscript_indices[subscript_index]] == -1:
                        subscript_indices_to_input_index[current_subscript_indices[subscript_index]] = dim_index_in_preprocessed_input
                        dim_index_in_preprocessed_input += 1
                        #homogenized_input_dims[subscript_index] = input_dims[dim_index_in_original_input]
                    else:
                        pass
                # for transpose
                permutation = []
                for d in six.moves.range(len(subscript_indices_to_input_index)):
                    if subscript_indices_to_input_index[d] != -1:
                        permutation.append(subscript_indices_to_input_index[d])
                IsTransposeRequired = False
                TransposeRequiredNum = -1
                TransposeRequiredOrder = None
                outTransOrder = []
                input_shape = self.get_shape_by_name(op_.input[i])
                for j in six.moves.range(len(current_subscript_indices)):
                    if permutation[j] != j:
                        IsTransposeRequired = True
                        TransposeRequiredNum = i
                        TransposeRequiredOrder = permutation

                if IsTransposeRequired:
                    outTransOrder = current_subscript_indices
                    if len(outTransOrder) < len(subscript_indices_to_input_index):
                        for j in six.moves.range(len(subscript_indices_to_input_index)):
                            if j not in outTransOrder:
                                outTransOrder.insert(j,j)

                # for reshape
                inputs_reshape_dims = [mul_outshape[d] if subscript_indices_to_input_index[d] != -1 else 1 for d in six.moves.range(len(subscript_indices_to_input_index)) ]
                preprocessed_inputs_reshape_dims.append(inputs_reshape_dims)
                inputs_reshape_dims = []

            return IsTransposeRequired, TransposeRequiredNum, TransposeRequiredOrder, preprocessed_inputs_reshape_dims

        def get_out_permutation(output_subscript_indices):
            res_temp = []
            for i,value in enumerate(output_subscript_indices):
                if value != -1:
                    res_temp.append(value)
            out_permutation = sorted(range(len(res_temp)), key=lambda k: res_temp[k], reverse=False)
            return out_permutation

        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                left_shape = self.get_shape_by_name(op_.input[0])
                if len(op_.input) > 1:
                    right_shape = self.get_shape_by_name(op_.input[1])
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'equation':
                        equation = arg[i].str
                custom_str = 'nkctv,kvw->nctw'
                if equation == custom_str:
                    return self.split_Einsum_custom(op)

                # 1\ equation_preprocessor
                # ---1.1 Split expressions to get input and output expressions
                left_equation, right_equation, in_equation, out_equation = einsum_equation_preprocessor(op_, equation)
                # ---1.2 Parsing expressions, using alphabetic indexes to uniformly express expressions
                input_subscript_indices, output_subscript_indices, letter_to_index, mul_outshape, reduce_axes_subscript = einsum_uniform_equation_index(op_, left_equation, right_equation, in_equation, out_equation)
                # ---1.3 Parsing expressions to obtain input and output pre-processing information
                IsTransposeRequired, TransposeRequiredNum, in_permutation, preprocessed_inputs_reshape_dims = PreprocessInputs(op_, in_equation, input_subscript_indices, letter_to_index, mul_outshape)

                # 2\ create transpose
                # ---If there is a transposition calculation, the input is transposed first, and then the dimension is filled by interpolating 1 with reshape, and the output also needs transposition calculation
                if IsTransposeRequired:
                    mace_check(TransposeRequiredNum != 0,'False,if need insert transpose it must be second input')
                    new_right_transpose = self.create_Transpose_op(op_, op_name + '_right', op_.input[TransposeRequiredNum], in_permutation)
                    op_.input[1] = new_right_transpose
                    # cal out_permutation for transpose out
                    out_permutation = get_out_permutation(output_subscript_indices)

                # 3\ create reshape
                # ---Complement the two operands by inserting 1 to make them have the same dimension
                new_left_reshape = self.create_Reshape_op(op_, op_name + '_left', op_.input[0], preprocessed_inputs_reshape_dims[0])
                op_.input[0] = new_left_reshape
                left_shape = preprocessed_inputs_reshape_dims[0]
                if len(op_.input) > 1:
                    if IsTransposeRequired:
                        r_input = new_right_transpose
                    else:
                        r_input =  op_.input[1]

                    new_right_reshape = self.create_Reshape_op(op_, op_name + '_right', r_input, preprocessed_inputs_reshape_dims[1])
                    op_.input[1] = new_right_reshape
                    right_shape = preprocessed_inputs_reshape_dims[1]

                # 4\ create mul
                # ---Calculate the operand with mul after filling the dimension
                # ---If there is dimension collapse, it will be realized by reducesum
                # ---If the input is transposed, the output also needs transposition calculation
                opName = op_name + '_mul'
                opType = 'MUL'
                outputTensorName = opName + '_output'
                if reduce_axes_subscript != []:
                    mul_out = self.create_Eltwise_op(op_, opName, opType, op_.input[0], op_.input[1], outputTensorName, mul_outshape)
                    # 5\ create sum
                    inputName_list = []
                    inputShape_list = []
                    need_create_out = True
                    sum_dimensions_axis = reduce_axes_subscript
                    sum_dimensions_axis.reverse()
                    for i in six.moves.range(len(sum_dimensions_axis)):
                        opName = op_name + '_sum' + str(i)
                        if i == 0:
                            inputTensorName = mul_out
                            inputTensorShape = mul_outshape
                        else:
                            inputTensorName = inputName_list[i-1]
                            inputTensorShape = inputShape_list[i-1]

                        if i == (len(sum_dimensions_axis) - 1):
                            if IsTransposeRequired == False:
                                sum_out,out_shape = self.create_Sum_op(op_, opName, inputTensorName, sum_dimensions_axis[i], inputTensorShape, False, need_create_out = False)
                            else:
                                sum_out,out_shape = self.create_Sum_op(op_, opName, inputTensorName, sum_dimensions_axis[i], inputTensorShape, False)
                                self.create_Transpose_op(op_, op_name + '_out_permute', sum_out, out_permutation,inputTensorShape = out_shape, need_create_out = False)
                        else:
                            sum_out,out_shape = self.create_Sum_op(op_, opName, inputTensorName, sum_dimensions_axis[i], inputTensorShape, False)
                        inputName_list.append(sum_out)
                        inputShape_list.append(out_shape)

                else:
                    if IsTransposeRequired == False:
                        mul_out = self.create_Eltwise_op(op_, opName, opType, op_.input[0], op_.input[1], need_create_out = False)
                    else:
                        mul_out = self.create_Eltwise_op(op_, opName, opType, op_.input[0], op_.input[1],outputTensorName, mul_outshape)
                        self.create_Transpose_op(op_, op_name + '_out_permute', mul_out, out_permutation, inputTensorShape = mul_outshape,need_create_out = False)

                self.remove_op_with_name(op_name)

    def split_FullyConnected(self, op):
        # unTransposed
        op_name = op.name
        xi = op.input[0]
        weight = op.input[1]
        bias = op.input[2]
        input_shape = self.get_shape_by_name(xi)
        # find input op and input op input_shape
        if len(input_shape) == 4 and (input_shape[1] !=1 or input_shape[2] !=1):
           #case for lenet
           [n,c,h,w] = input_shape
           # add reshape
           op_reshape = self._SGSModel.op.add()
           op_reshape.name = op_name + '_reshape#1'
           op_reshape.type = 'RESHAPE'
           op_reshape.input.extend([xi])
           reshape_output_tensor_name = op_reshape.name + '_output_shape'
           reshape_output_tensor_data = [1,1,1,-1]#n,h,w,c
           reshape_output_tensor_shape = [4]
           self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                           mace_pb2.DT_INT32, reshape_output_tensor_data)
           op_reshape.input.extend([reshape_output_tensor_name])
           output_op_reshape = op_reshape.name + '_output'
           op_reshape.output.extend([output_op_reshape])
           op_reshape.output_shape.add()
           op_reshape.output_shape[0].dims.extend([1,n*c*h*w,1,1])#n,c,h,w
           self._maceOpArray = np.append(self._maceOpArray,op_reshape)

           #creat ori fullconnect conv
           op_conv = self._SGSModel.op.add()
           op_conv.name = op_name + '_conv#1'
           op_conv.type = 'CONV_2D'
           strides_arg = op_conv.arg.add()
           strides_arg.name = 'strides'
           strides_arg.ints.extend([1,1])
           padding_arg = op_conv.arg.add()
           padding_arg.name = 'padding_values'
           padding_arg.ints.extend([0,0,0,0])
           #add padding_type
           paddingType_arg = op_conv.arg.add()
           paddingType_arg.name = MaceKeyword.mace_padding_str
           #set padding_type
           paddingType_arg.i = self.pooling_paddingType['CAFFE']

           tensor_weight = self.find_tensor_by_name(weight)
           temp_data = np.array(tensor_weight.float_data).reshape(op.output_shape[0].dims[1],n,c,h,w)
           temp_data = temp_data.transpose(0,1,3,4,2)
           del tensor_weight.float_data[:]
           tensor_weight.float_data.extend(list(temp_data.flat))
           op_conv.input.extend([output_op_reshape])
           op_conv.input.extend([weight])
           op_conv.input.extend([bias])
           output_op_conv = op_conv.name + '_output'
           op_conv.output[:] = op.output[:]
           op_conv.output_shape.extend(op.output_shape)
           #op_conv.output_shape[0].dims.extend([c,1,h,w])
           self._maceOpArray = np.append(self._maceOpArray,op_conv)
           self.remove_op_with_name(op_name)
        else:
             for op_ in self._SGSModel.op:
               if op_ == op:
                 op_.type = "FULLY_CONNECTED"
                 self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Gather(self, op):
        # unTransposed
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_name = op.name
            data_shape = self.get_shape_by_name (op_.input[0])
            indics_shape = self.get_shape_by_name (op_.input[1])
            output_shape = self.get_shape_by_name (op_.output[0])
            arg = op_.arg
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == MaceKeyword.mace_axis_str:
                axis = arg[i].i
            # Negative indices correspond to the innermost dimension of the input,
            # needs to be modified to a positive number and no need to transpose,
            # because input will be transposed back to onnx.
            indics = op_.input[1]
            indicsTensor = self.find_tensor_by_name(indics)
            if self.is_const_tensor(indics) == True:
                if indicsTensor.int32_data == [-1]:
                    indicsTensor.int32_data.pop()
                    new_indics = data_shape[axis] - 1
                    indicsTensor.int32_data.extend([new_indics])
                elif indicsTensor.float_data == [-1]:
                    indicsTensor.float_data.pop()
                    new_indics = data_shape[axis] - 1
                    indicsTensor.float_data.extend([new_indics])
            '''
            if len(input_shape) == len(weight_shape):
              # creat reshape to change x dim to 1 dim
              op_output_shape = [1]
              for i in six.moves.range(len(input_shape)):
                  op_output_shape[0] *= input_shape[i]
              op_reshape = self._SGSModel.op.add()
              op_reshape.name = op_.name + '_reshape'
              op_reshape.type = 'RESHAPE'
              reshape_output_tensor_name = op_reshape.name + '_output_shape'
              reshape_output_tensor_data = op_output_shape
              reshape_output_tensor_shape = [len(op_output_shape)]
              self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                           mace_pb2.DT_INT32, reshape_output_tensor_data)
              op_reshape.input.extend([op_.input[1]])
              op_reshape.input.extend([reshape_output_tensor_name])
              op_reshape_output_name = op_reshape.name + '_output'
              op_reshape.output.extend([op_reshape_output_name])
              op_reshape.output_shape.add()
              op_reshape.output_shape[0].dims.extend(op_output_shape)
              self._maceOpArray = np.append(self._maceOpArray,op_reshape)
              op_.input[1] = op_reshape_output_name
            '''

            op_.type = "GATHER"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_HardSigmoid(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_name = op_.name
            op_shape = self.get_shape_by_name(op_.input[0])
            op_out_shape = self.get_shape_by_name(op_.output[0])
            mace_check(len(op_shape) == len(op_out_shape),
                'HardSigmoid: {} need input and output same shape!'.format(op_name))
            for idx, shape in enumerate(op_shape):
                mace_check(shape == op_out_shape[idx],
                    'HardSigmoid: {} need input and output same shape!'.format(op_name))
            self.OpInputInsertTranspose(op_)
            input_shape = self.get_shape_by_name(op_.input[0])
            alpha = 0.2
            beta = 0.5
            for _, arg in enumerate(op_.arg):
                if arg.name == 'alpha':
                    alpha = arg.f
                if arg.name == 'beta':
                    beta = arg.f

            #                #
            # creat Mul      #
            #                #
            op_mul = self._SGSModel.op.add()
            op_mul.name = op_name + '_MUL'
            op_mul.type = 'MUL'
            # 1.add mul const tensor
            mul_const_name = op_name + '_MUL_const'
            mul_const_data = [alpha]
            mul_const_shape = [1]
            self.add_tensor(self._SGSModel, mul_const_name, mul_const_shape,
                mace_pb2.DT_FLOAT, mul_const_data)
            # 2.add mul output tensor
            op_mul.input.extend([op_.input[0]])
            op_mul.input.extend([mul_const_name])
            mul_output_name = op_name + '_MUL_output'
            op_mul.output.extend([mul_output_name])
            op_mul.output_shape.add()
            op_mul.output_shape[0].dims.extend(input_shape)
            self._maceOpArray = np.append(self._maceOpArray, op_mul)

            #                #
            # creat Add      #
            #                #
            op_add = self._SGSModel.op.add()
            op_add.name = op_name + '_ADD'
            op_add.type = 'ADD'
            # 1.add add const tensor
            add_const_name = op_name + '_ADD_const'
            add_const_data = [beta]
            add_const_shape = [1]
            self.add_tensor(self._SGSModel, add_const_name, add_const_shape,
                mace_pb2.DT_FLOAT, add_const_data)
            # 2.add add output tensor
            op_add.input.extend([mul_output_name])
            op_add.input.extend([add_const_name])
            add_output_name = op_name + '_ADD_output'
            op_add.output.extend([add_output_name])
            op_add.output_shape.add()
            op_add.output_shape[0].dims.extend(input_shape)
            self._maceOpArray = np.append(self._maceOpArray, op_add)

            #                #
            # creat Minimum  #
            #                #
            op_min = self._SGSModel.op.add()
            op_min.name = op_name + '_MIN'
            op_min.type = 'MINIMUM'
            # 1.add min const tensor
            min_const_name = op_name + '_MIN_const'
            min_const_data = [1]
            min_const_shape = [1]
            self.add_tensor(self._SGSModel, min_const_name, min_const_shape,
                mace_pb2.DT_FLOAT, min_const_data)
            # 2.add min output tensor
            op_min.input.extend([add_output_name])
            op_min.input.extend([min_const_name])
            min_output_name = op_name + '_MIN_output'
            op_min.output.extend([min_output_name])
            op_min.output_shape.add()
            op_min.output_shape[0].dims.extend(input_shape)
            self._maceOpArray = np.append(self._maceOpArray, op_min)

            #                #
            # creat Relu     #
            #                #
            op_relu = self._SGSModel.op.add()
            op_relu.name = op_name + '_RELU'
            op_relu.type = 'RELU'
            op_relu.input.extend([min_output_name])
            op_relu.output.extend([op_.output[0]])
            op_relu.output_shape.add()
            if len(op_out_shape) == 4:
               out_shape = [op_out_shape[0],op_out_shape[2],op_out_shape[3],op_out_shape[1]]
            op_relu.output_shape[0].dims.extend(op_out_shape)
            self._maceOpArray = np.append(self._maceOpArray, op_relu)
            self.OpOutputInsertTranspose(op_relu)
            self.remove_op_with_name(op_name)

    def split_InstanceNorm(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                input_tensor_name = op_.input[0]
                scale_tensor_name = op_.input[1]
                bias_tensor_name = op_.input[2]
                arg = op_.arg
                op_name = op_.name

                input_tensor = self.find_tensor_by_name(input_tensor_name)
                scale_tensor = self.find_tensor_by_name(scale_tensor_name)
                bias_tensor = self.find_tensor_by_name(bias_tensor_name)
                input_shape = self.get_shape_by_name(input_tensor_name) # NHWC
                scale_shape = self.get_shape_by_name(scale_tensor_name) # NHWC
                bias_shape = self.get_shape_by_name(bias_tensor_name)   # NHWC

                mace_check(input_shape == scale_shape or self.is_const_tensor(scale_tensor_name),
                    'InstanceNorm scale tensor must be constant, if need broadcast')
                mace_check(input_shape == bias_shape or self.is_const_tensor(bias_tensor_name),
                    'InstanceNorm bias tensor must be constant, if need broadcast')

                axis_tensor_name = op_name + '_axis'
                axis_tensor_data = [x for x in range(1, len(input_shape)-1)]
                axis_tensor_shape = [len(input_shape) - 2]
                self.add_tensor(self._SGSModel ,axis_tensor_name, axis_tensor_shape, mace_pb2.DT_INT32, axis_tensor_data)

                op_.input[1] = axis_tensor_name
                op_.input[2] = scale_tensor_name
                op_.input.extend([bias_tensor_name])

                if len(input_shape) == 4:
                    op_.type = 'CUSTOM'
                    op_.name = 'InstanceNorm' + op_name
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)
                else:
                    # ADD TRANSPOSE to change [N, C, D1, D2, ..., Dn] to [N, D1, D2, ..., Dn, C]
                    op_transposeIn = self._SGSModel.op.add()
                    op_transposeIn.name = op_name + '_transposeIn'
                    op_transposeIn.type = 'TRANSPOSE'
                    shape_tensor_name = op_transposeIn.name + '_shape'
                    shape_tensor_data = [x for x in range(0, len(input_shape))]
                    shape_tensor_data.append(shape_tensor_data[1])
                    del shape_tensor_data[1]
                    shape_tensor_shape = [len(input_shape)]
                    self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape,
                        mace_pb2.DT_INT32, shape_tensor_data)
                    op_transposeIn.input.extend([input_tensor_name])
                    op_transposeIn.input.extend([shape_tensor_name])
                    output_op_transposeIn = op_transposeIn.name + '_output'
                    op_transposeIn.output.extend([output_op_transposeIn])
                    tmp_shape = copy.deepcopy(input_shape)
                    tmp_shape.append(tmp_shape[1])
                    del tmp_shape[1]
                    op_transposeIn.output_shape.add()
                    op_transposeIn.output_shape[0].dims.extend(tmp_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transposeIn)

                    # ADD INSTANCE_NORM
                    op_instancenorm = self._SGSModel.op.add()
                    op_instancenorm.name = 'InstanceNorm' + op_name
                    op_instancenorm.type = 'CUSTOM'
                    op_instancenorm.input.extend([output_op_transposeIn])
                    op_instancenorm.input.extend([axis_tensor_name])
                    op_instancenorm.input.extend([scale_tensor_name])
                    op_instancenorm.input.extend([bias_tensor_name])
                    output_op_instancenorm = op_instancenorm.name + '_output'
                    op_instancenorm.output.extend([output_op_instancenorm])
                    op_instancenorm.output_shape.add()
                    op_instancenorm.output_shape[0].dims.extend(tmp_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_instancenorm)

                    # ADD TRANSPOSE to change [N, D1, D2, ..., Dn, C] to [N, C, D1, D2, ..., Dn]
                    op_transposeOut = self._SGSModel.op.add()
                    op_transposeOut.name = op_name + '_transposeOut'
                    op_transposeOut.type = 'TRANSPOSE'
                    shape_tensor_name = op_transposeOut.name + '_shape'
                    self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape,
                        mace_pb2.DT_INT32, shape_tensor_data)
                    op_transposeOut.input.extend([output_op_instancenorm])
                    op_transposeOut.input.extend([shape_tensor_name])
                    op_transposeOut.output.extend([op_.output[0]])
                    op_transposeOut.output_shape.add()
                    op_transposeOut.output_shape[0].dims.extend(input_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transposeOut)
                    self.remove_op_with_name(op_name)

    def split_GRU(self,op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                xi = op_.input[0]
                xi_shape = self.get_shape_by_name(xi)
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'gru_type':
                        gru_type = arg[i].i
                gru_input_arrays = []
                gru_data_input_tensor = xi
                h0_input = None

                def _Add_Reshape_For_Input(op_name, input, input_shape):
                  # add reshape
                  op_reshape = self._SGSModel.op.add()
                  op_reshape.name = op_name + '_reshape'
                  op_reshape.type = 'RESHAPE'
                  op_reshape.input.extend([input])

                  #tmp = shape_count//xi_shape[0]
                  reshape_output_tensor_name = op_reshape.name + '_output_shape'
                  reshape_output_tensor_data = [input_shape[0],1,input_shape[1],input_shape[2]]#n,h,w,c
                  reshape_output_tensor_shape = [4]
                  self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                  mace_pb2.DT_INT32, reshape_output_tensor_data)
                  op_reshape.input.extend([reshape_output_tensor_name])
                  output_op_reshape = op_reshape.name + '_output'
                  op_reshape.output.extend([output_op_reshape])
                  op_reshape.output_shape.add()
                  op_reshape.output_shape[0].dims.extend([input_shape[0],1,input_shape[1],input_shape[2]])#n,c,h,w
                  self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                  return output_op_reshape
                  #lstm_data_input_tensor = output_op_reshape


                '''
                reshpe lstm outputs from 4dims to output's ori dims
                '''
                def _Add_Reshape_For_Output(op_name, input_name, ori_shape, output_name):
                  # add reshape
                  op_reshape = self._SGSModel.op.add()
                  op_reshape.name = op_name + '_reshape'
                  op_reshape.type = 'RESHAPE'
                  op_reshape.input.extend([input_name])

                  #tmp = shape_count//xi_shape[0]
                  reshape_output_tensor_name = op_reshape.name + '_output_shape'
                  reshape_output_tensor_data = ori_shape
                  reshape_output_tensor_shape = [len(ori_shape)]
                  self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                  mace_pb2.DT_INT32, reshape_output_tensor_data)
                  op_reshape.input.extend([reshape_output_tensor_name])
                  output_op_reshape = output_name
                  op_reshape.output.extend([output_op_reshape])
                  op_reshape.output_shape.add()
                  op_reshape.output_shape[0].dims.extend(ori_shape)#n,c,h,w
                  self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                #chang inputs from 3 dims to 4 dims to fit sgs rule
                if len(xi_shape) != 4 or xi_shape[1] !=1 or xi_shape[2] !=1:
                    gru_data_input_tensor = _Add_Reshape_For_Input(op_name, xi, xi_shape)
                if (len(xi_shape) == 4 and xi_shape[3] !=1) or (len(xi_shape) == 4 and xi_shape[1] !=1):
                    gru_data_input_tensor = _Add_Reshape_For_Input(op_name, xi, xi_shape)

                if gru_type == 1:#h0 c0 are model inputs
                    h0_shape =  self.get_shape_by_name(op_.input[5])
                    if len(h0_shape) != 4:
                        h0_input = _Add_Reshape_For_Input(op_.input[5], op_.input[5], h0_shape)

                #single gru
                def _Create_GRU(op, input, h0, is_revers_part=False, is_forward_part=False):
                    '''
                        normal LSTM:
                        input       input  h0
                          |           |----|
                          |             |
                        gru    or      gru
                          |             |
                        concat        |---|
                          |           |   |
                                   concat hn

                    ================================
                        bi gru:

                        |          stridSlice
                        |             |
                       gru      +    gru    or    coming soon
                        |             |
                      concat        concat
                        |             |
                        |          stridSlice
                        |-------------|
                               |
                             concat
                               |

                    return last tensor(s)
                    '''
                    xi = op.input[0]
                    xi_shape = self.get_shape_by_name(xi)
                    if is_revers_part:
                        name_prefix = "sgs_subnet_reverse_gru" + str(self._gru_num)
                        reverse_input_tensor = gru_data_input_tensor

                        #reverse
                        op_slice = self._SGSModel.op.add()
                        op_slice.name = name_prefix + '_reverse_slice'
                        op_slice.type = 'STRIDED_SLICE'
                        op_slice.input.extend([reverse_input_tensor])
                        #tmp = shape_count//xi_shape[0]

                        # create new begin/end/stride tensors
                        strided_slice_begin_name = op_slice.name + '_begin'
                        strided_slice_begin_data = [-1,0,0,0]
                        self.add_tensor(self._SGSModel, strided_slice_begin_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_begin_data)
                        op_slice.input.extend([strided_slice_begin_name])

                        strided_slice_end_name = op_slice.name + '_end'
                        strided_slice_end_data = [-xi_shape[0]-1,1,xi_shape[1],xi_shape[2]]#n,c,h,w
                        self.add_tensor(self._SGSModel, strided_slice_end_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_end_data)
                        op_slice.input.extend([strided_slice_end_name])

                        strided_slice_strides_name = op_slice.name + '_strides'
                        strided_slice_strides_data = [-1,1,1,1]
                        self.add_tensor(self._SGSModel, strided_slice_strides_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_strides_data)
                        op_slice.input.extend([strided_slice_strides_name])

                        output_op_reverse_slice = op_slice.name + '_output'
                        op_slice.output.extend([output_op_reverse_slice])
                        op_slice.output_shape.add()
                        op_slice.output_shape[0].dims.extend([xi_shape[0],1,xi_shape[1],xi_shape[2]])#n,c,h,w
                        self._maceOpArray = np.append(self._maceOpArray,op_slice)

                        gru_input = output_op_reverse_slice

                    elif is_forward_part:
                        name_prefix = "sgs_subnet_forward_gru" + str(self._gru_num)
                        gru_input = input
                    else:
                        name_prefix = "sgs_subnet_gru" + str(self._gru_num)
                        gru_input = input

                    num_output = op.output_shape[0].dims[-1]
                    #add indicator tensor
                    t = op.output_shape[0].dims[0]
                    T_time = t
                    indicator_shape = [t,1]
                    indicator_data = np.ones(t,dtype=np.float32).reshape(t,1)
                    if gru_type == 0:
                        indicator_data[0,0] = 0.0
                    indicator_tensor_name = name_prefix + '_indicator'
                    self.add_tensor(self._SGSModel,indicator_tensor_name, indicator_data.shape,
                                    mace_pb2.DT_FLOAT,
                                    indicator_data.flatten().tolist())
                    # add h0
                    h0_name = name_prefix + '_h0'
                    h0_data = np.zeros((1,1,xi_shape[-2],num_output),dtype=np.float32)
                    h0_shape = [1,1,xi_shape[-2],num_output]
                    self.add_tensor(self._SGSModel, h0_name, h0_shape,
                                    mace_pb2.DT_FLOAT, h0_data.flatten().tolist())

                    # add offlie_size
                    offline_size_name = name_prefix + '_size'
                    offline_size_data = np.ones(2)
                    offline_size_shape = [2]
                    self.add_tensor(self._SGSModel, offline_size_name, offline_size_shape,
                                    mace_pb2.DT_FLOAT, offline_size_data)

                    #creat name-shape map for save to file
                    input_output_map = {}
                    input_output_map[name_prefix + '_input'] = [1,1,xi_shape[-2],xi_shape[-1]]
                    input_output_map[h0_name] = [1,1,xi_shape[-2],num_output]
                    input_output_map[name_prefix + '_time'] = [1,1]
                    input_output_map[name_prefix + '_output'] = op.output_shape[0].dims[:]
                    file_name = './gru_data/input_output_shape#' + str(self._gru_num) + '.npy'
                    self._gru_num += 1
                    np.save(file_name,input_output_map, allow_pickle=True)

                    #                #
                    # creat SGS_GRU #
                    #                #
                    op_SGS_GRU = self._SGSModel.op.add()
                    op_SGS_GRU.name = 'SGS_GRU' + op_name
                    op_SGS_GRU.type = 'CUSTOM'
                    gru_type_arg = op_SGS_GRU.arg.add()
                    gru_type_arg.name = 'gru_type'
                    gru_type_arg.i = gru_type

                    #add inputs
                    op_SGS_GRU.input.extend([gru_input])
                    op_SGS_GRU.input.extend([indicator_tensor_name])
                    if gru_type == 1:#h0 is model input
                        op_SGS_GRU.input.extend([h0_input])
                    else:
                        op_SGS_GRU.input.extend([h0_name])

                    op_SGS_GRU.input.extend([offline_size_name])
                    #add outputs
                    SGS_GRU_output_array = []
                    single_shape = [1,1,1,1]
                    single_shape[3] = op.output_shape[0].dims[-1]
                    single_shape[2] = xi_shape[-2]

                    for i in six.moves.range(T_time):
                      tmp_out_name = name_prefix + '_output' + str(i)
                      op_SGS_GRU.output.extend([tmp_out_name])
                      op_SGS_GRU.output_shape.add()
                      op_SGS_GRU.output_shape[i].dims.extend(single_shape)
                      SGS_GRU_output_array.append(tmp_out_name)

                    if gru_type == 1:
                        #chang outputs from 4 dims to ori dims
                        cn_hn_ori_shape = self.get_shape_by_name(op.output[1])
                        hn_out_name = name_prefix + '_output' + str(T_time-1)
                        _Add_Reshape_For_Output(hn_out_name, hn_out_name, cn_hn_ori_shape, op.output[1])

                    self._maceOpArray = np.append(self._maceOpArray,op_SGS_GRU)


                    # creat concat
                    op_concat = self._SGSModel.op.add()
                    op_concat.name = name_prefix + '_concat'
                    op_concat.type = 'CONCATENATION'
                    axis_arg = op_concat.arg.add()
                    axis_arg.name = 'axis'
                    axis_arg.i = 0
                    for name in SGS_GRU_output_array:
                      op_concat.input.extend([name])
                    if is_forward_part or is_revers_part:
                        output_op_concat = op_concat.name + '_output'
                    else:
                        output_op_concat = op.output[0] #norm lstm
                    concat_shape = [1,1,1,1]
                    concat_shape[0] = op.output_shape[0].dims[0]
                    #concat_shape[1] = op.output_shape[0].dims[2]
                    concat_shape[2] = op.output_shape[0].dims[2]
                    concat_shape[3] = op.output_shape[0].dims[3]
                    op_concat.output.extend([output_op_concat])
                    op_concat.output_shape.add()
                    op_concat.output_shape[0].dims.extend(concat_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_concat)

                    if is_revers_part:
                        #reverse
                        op_slice = self._SGSModel.op.add()
                        op_slice.name = name_prefix + '_reverse_output_slice'
                        op_slice.type = 'STRIDED_SLICE'
                        op_slice.input.extend([output_op_concat])
                        #tmp = shape_count//xi_shape[0]

                        # create new begin/end/stride tensors
                        strided_slice_begin_name = op_slice.name + '_begin'
                        strided_slice_begin_data = [-1,0,0,0]
                        self.add_tensor(self._SGSModel, strided_slice_begin_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_begin_data)
                        op_slice.input.extend([strided_slice_begin_name])

                        strided_slice_end_name = op_slice.name + '_end'
                        strided_slice_end_data = [-xi_shape[0]-1,1,xi_shape[1],op.output_shape[0].dims[3]]#n,h,w,c
                        self.add_tensor(self._SGSModel, strided_slice_end_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_end_data)
                        op_slice.input.extend([strided_slice_end_name])

                        strided_slice_strides_name = op_slice.name + '_strides'
                        strided_slice_strides_data = [-1,1,1,1]
                        self.add_tensor(self._SGSModel, strided_slice_strides_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_strides_data)
                        op_slice.input.extend([strided_slice_strides_name])

                        output_op_reverse_output_slice = op_slice.name + '_output'
                        output_slice_shape =  concat_shape #n,c,h,w
                        op_slice.output.extend([output_op_reverse_output_slice])
                        op_slice.output_shape.add()
                        op_slice.output_shape[0].dims.extend(output_slice_shape)#n,c,h,w
                        self._maceOpArray = np.append(self._maceOpArray,op_slice)
                        return output_op_reverse_output_slice
                    elif is_forward_part:
                        return output_op_concat
                    else:#norm lstm,output is ori output
                        outputTensor = self.find_tensor_by_name(output_op_concat)
                        return output_op_concat

                #norm gru
                if op_.output_shape[0].dims[1] == 1: # output wont be changed
                    gru_op_name = _Create_GRU (op_, gru_data_input_tensor, h0_input)
                # bi gru
                if op_.output_shape[0].dims[1] == 2:
                    forward_part = _Create_GRU (op_, gru_data_input_tensor, h0_input, False, True)
                    reverse_part = _Create_GRU (op_, gru_data_input_tensor, h0_input, True, False)

                    # creat total concat
                    op_concat = self._SGSModel.op.add()
                    op_concat.name = op_name + 'total_concat'
                    op_concat.type = 'CONCATENATION'
                    axis_arg = op_concat.arg.add()
                    axis_arg.name = 'axis'
                    axis_arg.i = 1

                    op_concat.input.extend([forward_part])
                    op_concat.input.extend([reverse_part])
                    output_op_concat_total = op_.output[0]
                    concat_shape = [1,1,1,1]
                    concat_shape = op_.output_shape[0].dims
                    op_concat.output.extend([output_op_concat_total])
                    op_concat.output_shape.add()
                    op_concat.output_shape[0].dims.extend(concat_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_concat)
                    outputTensor = self.find_tensor_by_name(output_op_concat_total)
                    gru_op_name = op_concat
                self.remove_op_with_name(op_name)

    def split_LSTM(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                #self.OpInputInsertTranspose(op_) # 4dim input,nhwc
                op_name = op_.name
                xi = op_.input[0]
                xi_shape = self.get_shape_by_name(xi)
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'lstm_type':
                        lstm_type = arg[i].i
                lstm_input_arrays = []
                lstm_data_input_tensor = xi
                h0_input = None
                c0_input = None
                if lstm_type == 1:#h0 c0 are model inputs
                    h0_input = op_.input[5]
                    c0_input = op_.input[6]
                '''
                reshpe lstm inputs from 3dims to 4dims
                '''
                def _Add_Reshape_For_Input(op_name, input, input_shape):
                    # add reshape
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape'
                    op_reshape.type = 'RESHAPE'
                    op_reshape.input.extend([input])

                    #tmp = shape_count//xi_shape[0]
                    reshape_output_tensor_name = op_reshape.name + '_output_shape'
                    reshape_output_tensor_data = [input_shape[0],1,input_shape[1],input_shape[2]]#n,h,w,c
                    reshape_output_tensor_shape = [4]
                    self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_output_tensor_data)
                    op_reshape.input.extend([reshape_output_tensor_name])
                    output_op_reshape = op_reshape.name + '_output'
                    op_reshape.output.extend([output_op_reshape])
                    op_reshape.output_shape.add()
                    op_reshape.output_shape[0].dims.extend([input_shape[0],1,input_shape[1],input_shape[2]])#n,c,h,w
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                    return output_op_reshape
                    #lstm_data_input_tensor = output_op_reshape

                '''
                reshpe lstm outputs from 4dims to output's ori dims
                '''
                def _Add_Reshape_For_Output(op_name, input_name, ori_shape, output_name):
                    # add reshape
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape'
                    op_reshape.type = 'RESHAPE'
                    op_reshape.input.extend([input_name])

                    #tmp = shape_count//xi_shape[0]
                    reshape_output_tensor_name = op_reshape.name + '_output_shape'
                    reshape_output_tensor_data = ori_shape
                    reshape_output_tensor_shape = [len(ori_shape)]
                    self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_output_tensor_data)
                    op_reshape.input.extend([reshape_output_tensor_name])
                    output_op_reshape = output_name
                    op_reshape.output.extend([output_op_reshape])
                    op_reshape.output_shape.add()
                    op_reshape.output_shape[0].dims.extend(ori_shape)#n,c,h,w
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                #chang inputs from 3 dims to 4 dims to fit sgs rule
                if len(xi_shape) != 4:
                    lstm_data_input_tensor = _Add_Reshape_For_Input(op_name, xi, xi_shape)
                if (len(xi_shape) == 4 and xi_shape[3] !=1) or (len(xi_shape) == 4 and xi_shape[1] !=1):
                    lstm_data_input_tensor = _Add_Reshape_For_Input(op_name, xi, xi_shape)
                if lstm_type == 1:#h0 c0 are model inputs
                    h0_shape =  self.get_shape_by_name(op_.input[5])
                    if len(h0_shape) != 4:
                        h0_input = _Add_Reshape_For_Input(op_.input[5], op_.input[5], h0_shape)
                    c0_shape = self.get_shape_by_name(op_.input[6])
                    if len(c0_shape) != 4:
                        c0_input = _Add_Reshape_For_Input(op_.input[6], op_.input[6], c0_shape)

                #single lstm
                def _Creat_LSTM(op, input, h0, c0, is_revers_part=False, is_forward_part=False):
                    '''
                        normal LSTM:
                        input    input  h0   c0
                          |        |----|-----|
                          |             |
                        lstm   or     lstm
                          |             |
                        concat      |---|---|
                          |         |   |   |
                                concat  hn  cn
                    ================================
                        bi LSTM:

                    strideSlice       |
                        |             |
                       lstm      +   lstm    or    coming soon
                        |             |
                      concat        concat
                        |             |
                    stridSlice        |
                        |-------------|
                               |
                             concat
                               |

                    return last tensor(s)
                    '''
                    xi = op.input[0]
                    xi_shape = self.get_shape_by_name(xi)
                    if is_revers_part:
                        name_prefix = "sgs_subnet_reverse_lstm" + str(self._lstm_num)
                        reverse_input_tensor = lstm_data_input_tensor

                        #reverse
                        op_slice = self._SGSModel.op.add()
                        op_slice.name = name_prefix + '_reverse_slice'
                        op_slice.type = 'STRIDED_SLICE'
                        op_slice.input.extend([reverse_input_tensor])
                        #tmp = shape_count//xi_shape[0]

                        # create new begin/end/stride tensors
                        strided_slice_begin_name = op_slice.name + '_begin'
                        strided_slice_begin_data = [-1,0,0,0]
                        self.add_tensor(self._SGSModel, strided_slice_begin_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_begin_data)
                        op_slice.input.extend([strided_slice_begin_name])

                        strided_slice_end_name = op_slice.name + '_end'
                        strided_slice_end_data = [-xi_shape[0]-1,1,xi_shape[1],xi_shape[2]]#n,c,h,w
                        self.add_tensor(self._SGSModel, strided_slice_end_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_end_data)
                        op_slice.input.extend([strided_slice_end_name])

                        strided_slice_strides_name = op_slice.name + '_strides'
                        strided_slice_strides_data = [-1,1,1,1]
                        self.add_tensor(self._SGSModel, strided_slice_strides_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_strides_data)
                        op_slice.input.extend([strided_slice_strides_name])

                        output_op_reverse_slice = op_slice.name + '_output'
                        op_slice.output.extend([output_op_reverse_slice])
                        op_slice.output_shape.add()
                        op_slice.output_shape[0].dims.extend([xi_shape[0],1,xi_shape[1],xi_shape[2]])#n,c,h,w
                        self._maceOpArray = np.append(self._maceOpArray,op_slice)

                        lstm_input = output_op_reverse_slice

                    elif is_forward_part:
                        name_prefix = "sgs_subnet_forward_lstm" + str(self._lstm_num)
                        lstm_input = input
                    else:
                        name_prefix = "sgs_subnet_lstm" + str(self._lstm_num)
                        lstm_input = input

                    num_output = op.output_shape[0].dims[-1]
                    #add indicator tensor
                    t = op.output_shape[0].dims[0]
                    T_time = t
                    indicator_shape = [t,1]
                    indicator_data = np.ones(t,dtype=np.float32).reshape(t,1)
                    if lstm_type == 0:
                        indicator_data[0,0] = 0.0
                    indicator_tensor_name = name_prefix + '_indicator'
                    self.add_tensor(self._SGSModel,indicator_tensor_name, indicator_data.shape,
                                    mace_pb2.DT_FLOAT,
                                    indicator_data.flatten().tolist())
                    # add h0
                    h0_name = name_prefix + '_h0'
                    h0_data = np.zeros((1,1,xi_shape[-2],num_output),dtype=np.float32)
                    h0_shape = [1,1,xi_shape[-2],num_output]
                    self.add_tensor(self._SGSModel, h0_name, h0_shape,
                                    mace_pb2.DT_FLOAT, h0_data.flatten().tolist())
                    # add c0
                    c0_name = name_prefix + '_c0'
                    c0_data = np.zeros((1,1,xi_shape[-2],num_output),dtype=np.float32)
                    c0_shape = [1,1,xi_shape[-2],num_output]
                    self.add_tensor(self._SGSModel, c0_name, c0_shape,
                                    mace_pb2.DT_FLOAT, c0_data.flatten().tolist())

                    # add offlie_size
                    offline_size_name = name_prefix + '_size'
                    offline_size_data = np.ones(2)
                    offline_size_shape = [2]
                    self.add_tensor(self._SGSModel, offline_size_name, offline_size_shape,
                                    mace_pb2.DT_FLOAT, offline_size_data)

                    #creat name-shape map for save to file
                    input_output_map = {}
                    input_output_map[name_prefix + '_input'] = [1,1,xi_shape[-2],xi_shape[-1]]
                    input_output_map[h0_name] = [1,1,xi_shape[-2],num_output]
                    input_output_map[c0_name] = [1,1,xi_shape[-2],num_output]
                    input_output_map[name_prefix + '_time'] = [1,1]
                    input_output_map[name_prefix + '_output'] = op.output_shape[0].dims[:]
                    file_name = './lstm_data/input_output_shape#' + str(self._lstm_num) + '.npy'
                    self._lstm_num += 1
                    np.save(file_name,input_output_map, allow_pickle=True)

                    #                #
                    # creat SGS_LSTM #
                    #                #
                    op_SGS_LSTM = self._SGSModel.op.add()
                    op_SGS_LSTM.name = 'SGS_LSTM' + op_name
                    op_SGS_LSTM.type = 'CUSTOM'
                    lstm_type_arg = op_SGS_LSTM.arg.add()
                    lstm_type_arg.name = 'lstm_type'
                    lstm_type_arg.i = lstm_type
                    #add inputs
                    op_SGS_LSTM.input.extend([lstm_input])
                    op_SGS_LSTM.input.extend([indicator_tensor_name])
                    if lstm_type == 1:#h0 c0 are model inputs
                        op_SGS_LSTM.input.extend([h0_input])
                        op_SGS_LSTM.input.extend([c0_input])
                    else:
                        op_SGS_LSTM.input.extend([h0_name])
                        op_SGS_LSTM.input.extend([c0_name])
                    op_SGS_LSTM.input.extend([offline_size_name])
                    #add outputs
                    SGS_LSTM_output_array = []
                    single_shape = [1,1,1,1]
                    single_shape[3] = op.output_shape[0].dims[-1]
                    #single_shape[1] = xi_shape[-2]
                    single_shape[2] = xi_shape[-2]

                    for i in six.moves.range(T_time):
                        tmp_out_name = name_prefix + '_output' + str(i)
                        op_SGS_LSTM.output.extend([tmp_out_name])
                        op_SGS_LSTM.output_shape.add()
                        op_SGS_LSTM.output_shape[i].dims.extend(single_shape)
                        SGS_LSTM_output_array.append(tmp_out_name)

                    if lstm_type == 1:
                        #add c_n-1
                        cn_out_name =  name_prefix + '_cn'
                        op_SGS_LSTM.output.extend([cn_out_name])
                        op_SGS_LSTM.output_shape.add()
                        op_SGS_LSTM.output_shape[T_time].dims.extend(single_shape)
                        #chang outputs from 4 dims to ori dims
                        cn_hn_ori_shape = self.get_shape_by_name(op.output[1])
                        hn_out_name = name_prefix + '_output' + str(T_time-1)
                        _Add_Reshape_For_Output(hn_out_name, hn_out_name, cn_hn_ori_shape, op.output[1])
                        _Add_Reshape_For_Output(cn_out_name, cn_out_name, cn_hn_ori_shape, op.output[2])

                    self._maceOpArray = np.append(self._maceOpArray,op_SGS_LSTM)
                    # creat concat
                    op_concat = self._SGSModel.op.add()
                    op_concat.name = name_prefix + '_concat'
                    op_concat.type = 'CONCATENATION'
                    axis_arg = op_concat.arg.add()
                    axis_arg.name = 'axis'
                    axis_arg.i = 0
                    for name in SGS_LSTM_output_array:
                        op_concat.input.extend([name])
                    if is_forward_part or is_revers_part:
                        output_op_concat = op_concat.name + '_output'
                    else:
                        output_op_concat = op.output[0] #norm lstm
                    concat_shape = [1,1,1,1]
                    concat_shape[0] = op.output_shape[0].dims[0]
                    #concat_shape[1] = op.output_shape[0].dims[2]
                    concat_shape[2] = op.output_shape[0].dims[2]
                    concat_shape[3] = op.output_shape[0].dims[3]
                    op_concat.output.extend([output_op_concat])
                    op_concat.output_shape.add()
                    op_concat.output_shape[0].dims.extend(concat_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_concat)
                    if is_revers_part:
                        #reverse
                        op_slice = self._SGSModel.op.add()
                        op_slice.name = name_prefix + '_reverse_output_slice'
                        op_slice.type = 'STRIDED_SLICE'
                        op_slice.input.extend([output_op_concat])
                        #tmp = shape_count//xi_shape[0]

                        # create new begin/end/stride tensors
                        strided_slice_begin_name = op_slice.name + '_begin'
                        strided_slice_begin_data = [-1,0,0,0]
                        self.add_tensor(self._SGSModel, strided_slice_begin_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_begin_data)
                        op_slice.input.extend([strided_slice_begin_name])

                        strided_slice_end_name = op_slice.name + '_end'
                        strided_slice_end_data = [-xi_shape[0]-1,1,xi_shape[1],op.output_shape[0].dims[3]]#n,h,w,c
                        self.add_tensor(self._SGSModel, strided_slice_end_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_end_data)
                        op_slice.input.extend([strided_slice_end_name])

                        strided_slice_strides_name = op_slice.name + '_strides'
                        strided_slice_strides_data = [-1,1,1,1]
                        self.add_tensor(self._SGSModel, strided_slice_strides_name, [4],
                                        mace_pb2.DT_INT32, strided_slice_strides_data)
                        op_slice.input.extend([strided_slice_strides_name])

                        output_op_reverse_output_slice = op_slice.name + '_output'
                        output_slice_shape =  concat_shape #n,c,h,w
                        op_slice.output.extend([output_op_reverse_output_slice])
                        op_slice.output_shape.add()
                        op_slice.output_shape[0].dims.extend(output_slice_shape)#n,c,h,w
                        self._maceOpArray = np.append(self._maceOpArray,op_slice)
                        return output_op_reverse_output_slice
                    elif is_forward_part:
                        return output_op_concat
                    else:#norm lstm,output is ori output
                        outputTensor = self.find_tensor_by_name(output_op_concat)
                        return output_op_concat


                #norm lstm
                if op_.output_shape[0].dims[1] == 1: # output wont be changed
                    lstm_op_name = _Creat_LSTM (op_, lstm_data_input_tensor, h0_input, c0_input)
                # bi lstm
                if op_.output_shape[0].dims[1] == 2:
                    forward_part = _Creat_LSTM (op_, lstm_data_input_tensor, h0_input, c0_input, False, True)
                    reverse_part = _Creat_LSTM (op_, lstm_data_input_tensor, h0_input, c0_input, True, False)

                    # creat total concat
                    op_concat = self._SGSModel.op.add()
                    op_concat.name = op_name + 'total_concat'
                    op_concat.type = 'CONCATENATION'
                    axis_arg = op_concat.arg.add()
                    axis_arg.name = 'axis'
                    axis_arg.i = 1

                    op_concat.input.extend([forward_part])
                    op_concat.input.extend([reverse_part])
                    output_op_concat_total = op_.output[0]
                    concat_shape = [1,1,1,1]
                    concat_shape = op_.output_shape[0].dims
                    op_concat.output.extend([output_op_concat_total])
                    op_concat.output_shape.add()
                    op_concat.output_shape[0].dims.extend(concat_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_concat)
                    outputTensor = self.find_tensor_by_name(output_op_concat_total)
                    lstm_op_name = op_concat
                self.remove_op_with_name(op_name)

    def split_BatchMatMul(self, op):
        # unTransposed
        # matmul A * B = C
        # Take the dimension of A as the benchmark
        #case 1: A:dims 2  B:dims 2
        #   switch 1:A:dims 2  variable tensor
        #            B:dims 2  variable tensor
        #   switch 2:A:dims 2  variable tensor
        #            B:dims 2  const tensor
        #
        #case 2: A:dims 3
        #   switch 1:A:dims 3  variable tensor
        #            B:dims 2  const tensor
        #   switch 2:A:dims 3  variable tensor
        #            B:dims 3  variable tensor
        #   switch 3:A:dims 3  variable tensor
        #            B:dims 2  variable tensor
        #   switch 3:A:dims 3  variable tensor
        #            B:dims 3  const tensor
        #
        #case 3: A:dims 4
        #   switch 1:A:dims 4  variable tensor
        #            B:dims 2  const tensor
        #   switch 2:A:dims 4  variable tensor
        #            B:dims 4  variable tensor
        #   switch 3:A:dims 4  variable tensor
        #            B:dims 4  const tensor
        #
        #case 4: A:dims > 4
        #   switch 1:A:dims > 4  variable tensor
        #            B:dims 2  const tensor
        #   switch 1:A:dims > 4  variable tensor
        #            B:dims > 4  variable tensor

        def transpose_2dimMatmul(op_, op_name,A,A_shape,B,B_shape,B_tensor, transA = False, transB = False, const_input_ori_data = None):
            if transA == True:
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '_transposeA'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shapeA'
                shape_tensor_data = [1,0]
                shape_tensor_shape = [2]
                self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([A])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_outputA'
                op_transpose.output.extend([output_op_transpose])
                tmp_shape = [A_shape[1],A_shape[0]]
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(tmp_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                op_.input[0] = output_op_transpose
            if transB == False:
                if self.is_const_tensor(B) == False:
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = 'SGS_' + op_name + '_transposeB'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shapeB'
                    shape_tensor_data = [1,0]
                    shape_tensor_shape = [2]
                    self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([B])
                    op_transpose.input.extend([shape_tensor_name])
                    output_op_transpose = op_transpose.name + '_outputB'
                    op_transpose.output.extend([output_op_transpose])
                    tmp_shape = [B_shape[1],B_shape[0]]
                    op_transpose.output_shape.add()
                    op_transpose.output_shape[0].dims.extend(tmp_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    self.remove_tensor_by_name(B_tensor)
                    op_.input[1] = output_op_transpose
                else:
                    self.handle_const(op_, "MATMUL", B, B_tensor, B_shape)
            return op_

        def change_2dimA_to_4dim(op, A_shape):
            op_ = op
            op_name = op.name
            A = op.input[0]

            # reshape A from 2dim to 4dim [N,C]->[N,C,1,1]
            op_output_shape = [A_shape[0],A_shape[1],1,1]
            op_reshape = self._SGSModel.op.add()
            op_reshape.name = op_name + '_reshapeA'
            op_reshape.type = 'RESHAPE'
            reshape_output_tensor_name = op_reshape.name + '_output_shapeA'
            reshape_output_tensor_data = op_output_shape
            reshape_output_tensor_shape = [len(op_output_shape)]
            self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                            mace_pb2.DT_INT32, reshape_output_tensor_data)
            op_reshape.input.extend([op_.input[0]])
            op_reshape.input.extend([reshape_output_tensor_name])
            op_reshape_output_name = op_reshape.name + '_outputA'
            op_reshape.output.extend([op_reshape_output_name])
            op_reshape.output_shape.add()
            op_reshape.output_shape[0].dims.extend(op_output_shape)
            self.add_tensor(self._SGSModel, op_reshape_output_name, op_output_shape,
                            mace_pb2.DT_FLOAT, None)
            self._maceOpArray = np.append(self._maceOpArray,op_reshape)
            #op_.input[0] = op_reshape_output_name

            # transpose A [N,C,1,1]->[N,1,1,C]
            op_transpose = self._SGSModel.op.add()
            op_transpose.name = 'SGS_' + op_name +'_transpose_2dim_A'
            op_transpose.type = 'TRANSPOSE'
            shape_tensor_name = op_transpose.name + '_shape'
            shape_tensor_data = [0,2,3,1]
            shape_tensor_shape = [4]
            self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                            mace_pb2.DT_INT32, shape_tensor_data)
            op_transpose.input.extend([op_reshape_output_name])
            op_transpose.input.extend([shape_tensor_name])
            output_op_transpose = op_transpose.name + '_output'
            op_transpose.output.extend([output_op_transpose])
            tmp_dim = [0,2,3,1]
            tmp_shape = [0,1,2,3]
            for i in six.moves.range(len(tmp_dim)):
                tmp_shape[i] = op_output_shape[tmp_dim[i]]
            op_transpose.output_shape.add()
            op_transpose.output_shape[0].dims.extend(tmp_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_transpose)

            # reshape A  [N,1,1,C]->[1,1,N,C]
            op_output_shape = [1,1,A_shape[0],A_shape[1]]
            op_reshape = self._SGSModel.op.add()
            op_reshape.name = op_name + '_reshapeA_2dim'
            op_reshape.type = 'RESHAPE'
            reshape_output_tensor_name = op_reshape.name + '_output_shapeA'
            reshape_output_tensor_data = op_output_shape
            reshape_output_tensor_shape = [len(op_output_shape)]
            self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                            mace_pb2.DT_INT32, reshape_output_tensor_data)
            op_reshape.input.extend([output_op_transpose])
            op_reshape.input.extend([reshape_output_tensor_name])
            op_reshape_output_name = op_reshape.name + '_outputA'
            op_reshape.output.extend([op_reshape_output_name])
            op_reshape.output_shape.add()
            op_reshape.output_shape[0].dims.extend(op_output_shape)
            self.add_tensor(self._SGSModel, op_reshape_output_name, op_output_shape,
                            mace_pb2.DT_FLOAT, None)
            self._maceOpArray = np.append(self._maceOpArray,op_reshape)

            return op_reshape_output_name

        def change_2dimB_to_4dim(op, B_shape, new_data = []):
            op_ = op
            op_name = op.name
            B = op.input[1]
            op_name = op.name
            if new_data != []:
                const_tensor_shape = [B_shape[0],1,1,B_shape[1]]
                const_tensor_name = op_name + '_reshape_constB'
                const_tensor_data = new_data
                self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                mace_pb2.DT_FLOAT, const_tensor_data)
                op_.input[1] = const_tensor_name
                return op_
            else:
                # reshape A from 2dim to 4dim [N,C]->[N,C,1,1]
                op_output_shape = [B_shape[0],B_shape[1],1,1]
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshapeB'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shapeB'
                reshape_output_tensor_data = op_output_shape
                reshape_output_tensor_shape = [len(op_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([op_.input[1]])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_outputB'
                op_reshape.output.extend([op_reshape_output_name])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(op_output_shape)
                self.add_tensor(self._SGSModel, op_reshape_output_name, op_output_shape,
                                mace_pb2.DT_FLOAT, None)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                #op_.input[0] = op_reshape_output_name

                # transpose A [N,C,1,1]->[N,1,1,C]
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name +'_transpose_2dim_B'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0,2,3,1]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([op_reshape_output_name])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                tmp_dim = [0,2,3,1]
                tmp_shape = [0,1,2,3]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = op_output_shape[tmp_dim[i]]
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(tmp_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                # reshape A  [N,1,1,C]->[1,1,N,C]
                op_output_shape = [1,1,B_shape[0],B_shape[1]]
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshapeB_2dim'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shapeB'
                reshape_output_tensor_data = op_output_shape
                reshape_output_tensor_shape = [len(op_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([output_op_transpose])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_outputB'
                op_reshape.output.extend([op_reshape_output_name])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(op_output_shape)
                self.add_tensor(self._SGSModel, op_reshape_output_name, op_output_shape,
                                mace_pb2.DT_FLOAT, None)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                return op_reshape_output_name


        # main
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                A = op_.input[0]              # left matrix name
                B = op_.input[1]              # right  matrix name
                C = op_.output[0]             # Matrix multiply results name from A * B
                A_tensor = self.find_tensor_by_name(A)
                B_tensor = self.find_tensor_by_name(B)
                C_tensor = self.find_tensor_by_name(C)
                A_shape = copy.deepcopy(A_tensor.dims)
                B_shape = copy.deepcopy(B_tensor.dims)
                C_shape = copy.deepcopy(self.find_tensor_by_name(C).dims) # list
                A_dim_num = len(A_tensor.dims)
                B_dim_num = len(B_tensor.dims)
                B_is_const = False
                if self.is_const_tensor(B) == True:
                    B_is_const = True
                    const_input_ori_data_B = self.get_const_ori_data(B)
                if self.is_const_tensor(A) == True:
                    return self.split_LhsIsConstMatmul(op_)
                # 2dim matmul
                if A_dim_num == 2:
                    # GEMM
                    # 1\ get tranpose Flag
                    transA = False
                    transB = False
                    arg = op_.arg
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_transpose_a_str:
                            transA = arg[i].i
                        elif name == MaceKeyword.mace_transpose_b_str:
                            transB = arg[i].i

                    # 2\create bias tensor
                    if len(op_.input) < 3:
                        bias_data = np.zeros(C_shape[-1])
                        bias_tensor_name = op_name + '_bias'
                        self.add_tensor(self._SGSModel,bias_tensor_name, bias_data.shape,
                                        mace_pb2.DT_FLOAT,bias_data)
                        op_.input.extend([bias_tensor_name])
                    else:
                        bias_tensor_name = op_.input[2]
                        bias_tensor = self.find_tensor_by_name(bias_tensor_name)
                        bias_shape = bias_tensor.dims
                        if bias_shape[-1] == 1 and bias_shape[-1] != C_shape[-1]:
                            bias_shape[-1] = C_shape[-1]
                            bias_data = self.get_const_ori_data(bias_tensor_name)
                            bias_data = np.tile(bias_data, C_shape[-1])
                            bias_tensor_name = op_name + '_bias'
                            self.add_tensor(self._SGSModel,bias_tensor_name, bias_shape,
                                             mace_pb2.DT_FLOAT,bias_data)
                            op_.input[2] = bias_tensor_name
                            del bias_tensor

                    # 3\transpose input according Flag
                    if self.is_const_tensor(B) == True:
                        op_ = transpose_2dimMatmul(op_,op_name,A,A_shape,B,B_shape,B_tensor,transA,transB,const_input_ori_data_B)
                    else:
                        op_ = transpose_2dimMatmul(op_,op_name,A,A_shape,B,B_shape,B_tensor,transA,transB)

                    # 4\2dim->4dim
                    ## 4.1\ change A
                    if transA == True:
                        A_shape = [A_shape[1],A_shape[0]]
                    NewInputTensor1 = change_2dimA_to_4dim(op_, A_shape)
                    ## 4.2\ change B
                    if transB == False:
                        B_shape = [B_shape[1],B_shape[0]]
                    else:
                        B_shape = B_shape
                    if B_is_const == True:
                        if transB == False:
                            new_data = list(((const_input_ori_data_B.reshape(B_tensor.dims)).transpose(1,0)).flat)
                        else:
                            new_data = const_input_ori_data_B
                        new_op = change_2dimB_to_4dim(op_, B_shape, new_data)
                        NewInputTensor2 = new_op.input[1]
                    else:
                        NewInputTensor2 = change_2dimB_to_4dim(op_, B_shape)

                    # 5\create BatchMatMul
                    op_fc = self._SGSModel.op.add()
                    op_fc.name = op_name + '_fc'
                    op_fc.type = 'FULLY_CONNECTED'
                    bias_tensor_name = op_.input[2]
                    op_fc.input.extend([NewInputTensor1])
                    op_fc.input.extend([NewInputTensor2])
                    op_fc.input.extend([bias_tensor_name])
                    output_op_fc = op_fc.name + '_output4dim'
                    op_fc.output.extend([output_op_fc])
                    op_fc.output_shape.add()
                    output_shape = [1, 1, C_shape[0], C_shape[1]]
                    op_fc.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_fc)

                    # 6\reshape output  [1,1,N,K]->[N,1,1,K]
                    op_output_shape = [C_shape[0], 1, 1, C_shape[1]]
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape2to4'
                    op_reshape.type = 'RESHAPE'
                    reshape_output_tensor_name = op_reshape.name + '_output'
                    reshape_output_tensor_data = op_output_shape
                    reshape_output_tensor_shape = [len(op_output_shape)]
                    self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_output_tensor_data)
                    op_reshape.input.extend([output_op_fc])
                    op_reshape.input.extend([reshape_output_tensor_name])
                    op_reshape_output_name = op_reshape.name + '_output2to4'
                    op_reshape.output.extend([op_reshape_output_name])
                    op_reshape.output_shape.add()
                    op_reshape.output_shape[0].dims.extend(op_output_shape)
                    self.add_tensor(self._SGSModel, op_reshape_output_name, op_output_shape,
                                    mace_pb2.DT_FLOAT, None)
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                    op_input_shape = op_output_shape
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = 'SGS_' + op_name + '_transposeOut'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shapeOut'
                    shape_tensor_data = [0,3,1,2]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([op_reshape_output_name])
                    op_transpose.input.extend([shape_tensor_name])
                    # 7\transpose output
                    tmp_dim = [0,3,1,2]
                    tmp_shape = [0,1,2,3]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = op_input_shape[tmp_dim[i]]
                    output_op_transpose = op_transpose.name + '_output4dim'
                    op_transpose.output.extend([output_op_transpose])
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(tmp_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                    op_output_shape = [C_shape[0],C_shape[1]]
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshapeC'
                    op_reshape.type = 'RESHAPE'
                    reshape_output_tensor_name = op_reshape.name + '_output_shapeC'
                    reshape_output_tensor_data =op_output_shape
                    reshape_output_tensor_shape = [len(op_output_shape)]
                    self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_output_tensor_data)
                    op_reshape.input.extend([output_op_transpose])
                    op_reshape.input.extend([reshape_output_tensor_name])
                    op_reshape.output.extend([C])
                    op_reshape.output_shape.add()
                    op_reshape.output_shape[0].dims.extend(op_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                    self.remove_op_with_name(op_name)

                # 3dim matmul
                elif A_dim_num == 3:
                    # A : 1*3*4   B : 1*4*3, B need to trans to same as A
                    if B_dim_num == 3:
                        if self.is_const_tensor(B) == False:
                            op_transpose = self._SGSModel.op.add()
                            op_transpose.name = 'SGS_' + op_name + '_transposeB'
                            op_transpose.type = 'TRANSPOSE'
                            shape_tensor_name = op_transpose.name + '_shapeB'
                            shape_tensor_data = [0,2,1]
                            shape_tensor_shape = [3]
                            self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                            op_transpose.input.extend([B])
                            op_transpose.input.extend([shape_tensor_name])
                            output_op_transpose = op_transpose.name + '_outputB'
                            op_transpose.output.extend([output_op_transpose])
                            tmp_dim = [0,2,1]
                            tmp_shape = [0,1,2]
                            for i in six.moves.range(len(tmp_dim)):
                                tmp_shape[i] = B_shape[tmp_dim[i]]
                            B_shape = tmp_shape
                            op_transpose.output_shape.add()
                            op_transpose_shape = copy.deepcopy(tmp_shape)
                            op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                            op_.input[1] = output_op_transpose
                        else:
                            self.handle_const(op_, "MATMUL", B, B_tensor, B_shape)

                    # A : 1*3*4   B : 4*3, B need to trans to same as A[-2:]
                    elif B_dim_num == 2:
                        if self.is_const_tensor(B) == False:
                            op_transpose = self._SGSModel.op.add()
                            op_transpose.name = 'SGS_' + op_name + '_transposeB'
                            op_transpose.type = 'TRANSPOSE'
                            shape_tensor_name = op_transpose.name + '_shapeB'
                            shape_tensor_data = [1,0]
                            shape_tensor_shape = [2]
                            self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                            op_transpose.input.extend([B])
                            op_transpose.input.extend([shape_tensor_name])
                            output_op_transpose = op_transpose.name + '_outputB'
                            op_transpose.output.extend([output_op_transpose])
                            tmp_dim = [1,0]
                            tmp_shape = [0,1]
                            for i in six.moves.range(len(tmp_dim)):
                                tmp_shape[i] = B_shape[tmp_dim[i]]
                            op_transpose.output_shape.add()
                            op_transpose_shape = copy.deepcopy(tmp_shape)
                            op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                            B_shape = tmp_shape
                            op_.input[1] = output_op_transpose
                        else:
                            self.handle_const(op_, "MATMUL", B, B_tensor, B_shape)

                    op_.type = "BATCH_MATMUL"
                    self._maceOpArray = np.append(self._maceOpArray, op_)
                # 4dim matmul
                elif A_dim_num == 4:
                    if B_dim_num == 4:
                        if self.is_const_tensor(B) == False:
                            op_transpose = self._SGSModel.op.add()
                            op_transpose.name = 'SGS_' + op_name + '_transposeB'
                            op_transpose.type = 'TRANSPOSE'
                            shape_tensor_name = op_transpose.name + '_shapeB'
                            shape_tensor_data = [0,1,3,2]
                            shape_tensor_shape = [4]
                            self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape,
                                            mace_pb2.DT_INT32, shape_tensor_data)
                            op_transpose.input.extend([B])
                            op_transpose.input.extend([shape_tensor_name])
                            output_op_transpose = op_transpose.name + '_outputB'
                            op_transpose.output.extend([output_op_transpose])
                            tmp_dim = [0,1,3,2]
                            tmp_shape = [0,1,2,3]
                            for i in six.moves.range(len(tmp_dim)):
                                tmp_shape[i] = B_shape[tmp_dim[i]]
                            op_transpose.output_shape.add()
                            op_transpose_shape = copy.deepcopy(tmp_shape)
                            op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                            self.add_tensor(self._SGSModel, output_op_transpose, op_transpose_shape,
                                            mace_pb2.DT_FLOAT, None)
                            self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                            B_shape = tmp_shape
                            op_.input[1] = output_op_transpose
                        else:
                            self.handle_const(op_, "MATMUL", B, B_tensor, B_shape)
                    elif B_dim_num == 2 and self.is_const_tensor(B) == True:
                        self.handle_const(op_, "MATMUL", B, B_tensor, B_shape)

                    op_.type = "BATCH_MATMUL"
                    self._maceOpArray = np.append(self._maceOpArray, op_)
                else:
                    mace_check(A_dim_num < 5 , "do not support more than 4 dims MatMul")

    def split_LhsIsConstMatmul(self, op):
        # matmul A * B = C
        # Take the dimension of A as the benchmark
        #case 1: A:dims 2  B:dims 2
        #   switch 1:A:dims 2  const tensor
        #            B:dims 2  variable tensor
        #
        #case 2: A:dims 3
        #   switch 1:A:dims 3  const tensor
        #            B:dims 2  variable tensor
        #   switch 2:A:dims 3  const tensor
        #            B:dims 3  variable tensor
        #
        #case 3: A:dims 4
        #   switch 1:A:dims 4  const tensor
        #            B:dims 2  variable tensor
        #   switch 2:A:dims 4  const tensor
        #            B:dims 4  variable tensor
        #
        #case 4: A:dims > 4
        #   switch 1:A:dims > 4  const tensor
        #            B:dims 2  variable tensor
        #   switch 1:A:dims > 4  const tensor
        #            B:dims > 4  variable tensor
        #
        # TODO  # swap left and right operands
                # A is const tensor, op.input[0] == A
                #                 -> op.input[1] == A
                # B is variable tensor, transpose B[-2], op.input[1] == B
                #                                     -> op.input[0] == NEW_B
                # transpose C[-2], op.output[0] == C
                #               -> op.output[0] == NEW_C

        def _split_2dim_LhsIsConstMatmul(op):
            def transpose_2dimMatmul(op_, op_name,A,A_shape,B,B_shape,B_tensor, transA = False, transB = False, const_input_ori_data = None):
                # 1. create new input[0]
                if transA == True:
                    self.handle_const(op_, "MATMUL", A, A_tensor, A_shape,index = 0)
                # 2. create new input[1]
                if transB == False:
                    if self.is_const_tensor(B) == False:
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = 'SGS_' + op_name + '_transposeB'
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shapeB'
                        shape_tensor_data = [1,0]
                        shape_tensor_shape = [2]
                        self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([B])
                        op_transpose.input.extend([shape_tensor_name])
                        output_op_transpose = op_transpose.name + '_outputB'
                        op_transpose.output.extend([output_op_transpose])
                        tmp_shape = [B_shape[1],B_shape[0]]
                        op_transpose.output_shape.add()
                        op_transpose.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                        op_.input[1] = output_op_transpose
                return op_

            def change_2dimA_to_4dim(op, A_shape,new_data = []):
                if new_data != []:
                    op_ = op
                    op_name = op.name
                    A = op.input[0]
                    A_shape = [A_shape[0],1,1,A_shape[1]]
                    const_tensor_name = op_name + '_reshape_constA'
                    const_tensor_shape = A_shape
                    const_tensor_data = new_data
                    self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                    mace_pb2.DT_FLOAT, const_tensor_data)
                    op_.input[0] = const_tensor_name
                    return op_

            def change_2dimB_to_4dim(op, B_shape):
                op_ = op
                op_name = op.name
                op_name = op.name

                # reshape A from 2dim to 4dim [N,C]->[N,C,1,1]
                op_output_shape = [B_shape[0],B_shape[1],1,1]
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshapeB'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shapeB'
                reshape_output_tensor_data = op_output_shape
                reshape_output_tensor_shape = [len(op_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([op.input[1]])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_outputB'
                op_reshape.output.extend([op_reshape_output_name])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(op_output_shape)
                self.add_tensor(self._SGSModel, op_reshape_output_name, op_output_shape,
                                mace_pb2.DT_FLOAT, None)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                #op_.input[0] = op_reshape_output_name

                # transpose A [N,C,1,1]->[N,1,1,C]
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name +'_transpose_2dim_B'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0,2,3,1]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([op_reshape_output_name])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                tmp_dim = [0,2,3,1]
                tmp_shape = [0,1,2,3]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = op_output_shape[tmp_dim[i]]
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(tmp_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                # reshape A  [N,1,1,C]->[1,1,N,C]
                op_output_shape = [1,1,B_shape[0],B_shape[1]]
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshapeB_2dim'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shapeB'
                reshape_output_tensor_data = op_output_shape
                reshape_output_tensor_shape = [len(op_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([output_op_transpose])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_outputB'
                op_reshape.output.extend([op_reshape_output_name])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(op_output_shape)
                self.add_tensor(self._SGSModel, op_reshape_output_name, op_output_shape,
                                mace_pb2.DT_FLOAT, None)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                return op_reshape_output_name

            for op_ in self._SGSModel.op:
                if op_ == op:
                    op_name = op_.name
                    A = op_.input[0]              # left matrix name
                    B = op_.input[1]              # right  matrix name
                    C = op_.output[0]             # Matrix multiply results name from A * B
                    A_tensor = self.find_tensor_by_name(A)
                    B_tensor = self.find_tensor_by_name(B)
                    C_tensor = self.find_tensor_by_name(C)
                    A_shape = A_tensor.dims
                    B_shape = B_tensor.dims
                    C_shape = self.find_tensor_by_name(C).dims # list
                    A_dim_num = len(A_tensor.dims)
                    B_dim_num = len(B_tensor.dims)
                    A_is_const = False
                    if self.is_const_tensor(A) == True:
                        A_is_const = True
                        const_input_ori_data_A = self.get_const_ori_data(A)

                    # GEMM
                    transA = False
                    transB = False
                    arg = op_.arg
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_transpose_a_str:
                            transA = arg[i].i
                        elif name == MaceKeyword.mace_transpose_b_str:
                            transB = arg[i].i
                    if len(op_.input) < 3:
                        bias_data = np.zeros(C_shape[-2])
                        bias_tensor_name = op_name + '_bias'
                        self.add_tensor(self._SGSModel,bias_tensor_name, bias_data.shape,
                                        mace_pb2.DT_FLOAT,bias_data)
                        op_.input.extend([bias_tensor_name])
                    else:
                        bias_tensor_name = op.input[2]
                        bias_tensor = self.find_tensor_by_name(bias_tensor_name)
                        bias_shape = bias_tensor.dims
                        if bias_shape[-1] == 1 and bias_shape[-1] != C_shape[-1]:
                            bias_shape[-1] = C_shape[-1]
                            bias_data = self.get_const_ori_data(self, bias_tensor_name)
                            bias_data = np.tile(bias_data, C_shape[-1])
                            bias_tensor_name = op_name + '_bias'
                            self.add_tensor(self._SGSModel,bias_tensor_name, bias_shape,
                                             mace_pb2.DT_FLOAT,bias_data)
                            op_.input[2] = bias_tensor_name
                            del bias_tensor
                    # 1、transpose input
                    if self.is_const_tensor(A) == True:
                        op_ = transpose_2dimMatmul(op_,op_name,A,A_shape,B,B_shape,B_tensor,transA,transB,const_input_ori_data_A)

                    # 2、2dim->4dim
                    ##  change A
                    if A_is_const == True:
                        if transA == True:
                            new_data = list(((const_input_ori_data_A.reshape(A_shape)).transpose(1,0)).flat)
                            A_shape = [A_shape[1],A_shape[0]]
                        else:
                            new_data = const_input_ori_data_A
                        op_ = change_2dimA_to_4dim(op_, A_shape, new_data)
                        NewInputTensor2 = op_.input[0]
                    ##  change B
                    if transB == False:
                        B_shape = [B_shape[1],B_shape[0]]
                    NewInputTensor1 = change_2dimB_to_4dim(op_, B_shape)

                    # 3、create BatchMatMul
                    op_fc = self._SGSModel.op.add()
                    op_fc.name = op_name + '_fc'
                    op_fc.type = 'FULLY_CONNECTED'
                    bias_tensor_name = op_.input[2]
                    op_fc.input.extend([NewInputTensor1])
                    op_fc.input.extend([NewInputTensor2])
                    op_fc.input.extend([bias_tensor_name])
                    output_op_fc = op_fc.name + '_output4dim'
                    op_fc.output.extend([output_op_fc])
                    op_fc.output_shape.add()
                    output_shape = [1, 1, C_shape[1], C_shape[0]]
                    op_fc.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_fc)

                    # 4\reshape output  [1,1,N,K]->[N,1,1,K]
                    op_output_shape = [C_shape[1], 1, 1, C_shape[0]]
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape2to4'
                    op_reshape.type = 'RESHAPE'
                    reshape_output_tensor_name = op_reshape.name + '_output'
                    reshape_output_tensor_data = op_output_shape
                    reshape_output_tensor_shape = [len(op_output_shape)]
                    self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_output_tensor_data)
                    op_reshape.input.extend([output_op_fc])
                    op_reshape.input.extend([reshape_output_tensor_name])
                    op_reshape_output_name = op_reshape.name + '_output2to4'
                    op_reshape.output.extend([op_reshape_output_name])
                    op_reshape.output_shape.add()
                    op_reshape.output_shape[0].dims.extend(op_output_shape)
                    self.add_tensor(self._SGSModel, op_reshape_output_name, op_output_shape,
                                    mace_pb2.DT_FLOAT, None)
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                    op_input_shape = op_output_shape
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = 'SGS_' + op_name + '_transposeOut'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shapeOut'
                    shape_tensor_data = [0,3,1,2]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([op_reshape_output_name])
                    op_transpose.input.extend([shape_tensor_name])

                    # 5\transpose output
                    tmp_dim = [0,3,1,2]
                    tmp_shape = [0,1,2,3]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = op_input_shape[tmp_dim[i]]
                    output_op_transpose = op_transpose.name + '_output4dim'
                    op_transpose.output.extend([output_op_transpose])
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(tmp_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                    # 6、4dim->2dim
                    op_output_shape = [C_shape[1],C_shape[0]]
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshapeC'
                    op_reshape.type = 'RESHAPE'
                    reshape_output_tensor_name = op_reshape.name + '_output_shapeC'
                    reshape_output_tensor_data =op_output_shape
                    reshape_output_tensor_shape = [len(op_output_shape)]
                    self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_output_tensor_data)
                    op_reshape.input.extend([output_op_transpose])
                    op_reshape.input.extend([reshape_output_tensor_name])
                    output_op_reshape = op_reshape.name + '_output4dim'
                    op_reshape.output.extend([output_op_reshape])
                    op_reshape.output_shape.add()
                    op_reshape.output_shape[0].dims.extend(op_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                    # 7、create new output[0]
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = 'SGS_' + op_name + '_transpose_3dim_output'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = [1,0]
                    shape_tensor_shape = [2]
                    self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([output_op_reshape])
                    op_transpose.input.extend([shape_tensor_name])
                    op_transpose.output.extend([C])
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(C_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    self.remove_op_with_name(op_name)

        def _split_3dim_LhsIsConstMatmul(op):
            for op_ in self._SGSModel.op:
                if op_ == op:
                    op_name = op_.name
                    A = op_.input[0]              # left matrix name
                    B = op_.input[1]              # right  matrix name
                    C = op_.output[0]             # Matrix multiply results name from A * B
                    A_tensor = self.find_tensor_by_name(A)
                    B_tensor = self.find_tensor_by_name(B)
                    C_tensor = self.find_tensor_by_name(C)
                    A_shape = A_tensor.dims
                    B_shape = B_tensor.dims
                    C_shape = self.find_tensor_by_name(C).dims # list
                    A_dim_num = len(A_tensor.dims)
                    B_dim_num = len(B_tensor.dims)
                    # 1. create new input[0]
                    if B_dim_num == 3:
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = 'SGS_' + op_name + '_transposeB'
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shapeB'
                        shape_tensor_data = [0,2,1]
                        shape_tensor_shape = [3]
                        self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([B])
                        op_transpose.input.extend([shape_tensor_name])
                        output_op_transpose = op_transpose.name + '_outputB'
                        op_transpose.output.extend([output_op_transpose])
                        tmp_dim = [0,2,1]
                        tmp_shape = [0,1,2]
                        for i in six.moves.range(len(tmp_dim)):
                            tmp_shape[i] = B_shape[tmp_dim[i]]
                        B_shape = tmp_shape
                        op_transpose.output_shape.add()
                        op_transpose_shape = copy.deepcopy(tmp_shape)
                        op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                        op_.input[0] = output_op_transpose
                    elif B_dim_num == 2:
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = op_name + '_transposeB'
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shapeB'
                        shape_tensor_data = [1,0]
                        shape_tensor_shape = [2]
                        self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([B])
                        op_transpose.input.extend([shape_tensor_name])
                        output_op_transpose = op_transpose.name + '_outputB'
                        op_transpose.output.extend([output_op_transpose])
                        tmp_dim = [1,0]
                        tmp_shape = [1,0]
                        for i in six.moves.range(len(tmp_dim)):
                            tmp_shape[i] = B_shape[tmp_dim[i]]
                        B_shape = tmp_shape
                        op_transpose.output_shape.add()
                        op_transpose_shape = copy.deepcopy(tmp_shape)
                        op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                        op_.input[0] = output_op_transpose

                    # 2. create new input[1]
                    op_.input[1] = A

                    # 3. create BatchMatMul
                    op_MatMul= self._SGSModel.op.add()
                    op_MatMul.name = op_name + '_BatchMatMul_3dim'
                    op_MatMul.type = "BATCH_MATMUL"
                    op_MatMul.input.extend([op_.input[0]])
                    op_MatMul.input.extend([op_.input[1]])
                    output_op_MatMul = op_MatMul.name + '_output'
                    op_MatMul.output.extend([output_op_MatMul])
                    op_MatMul.output_shape.add()
                    output_shape = [C_shape[0], C_shape[2], C_shape[1]]
                    op_MatMul.output_shape[0].dims.extend(output_shape)
                    self.add_tensor(self._SGSModel, output_op_MatMul, output_shape,
                                    mace_pb2.DT_FLOAT, None)
                    self._maceOpArray = np.append(self._maceOpArray,op_MatMul)
                    # 4. create new output[0]
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = op_name + '_transpose_3dim_output'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = [0,2,1]
                    shape_tensor_shape = [3]
                    self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([output_op_MatMul])
                    op_transpose.input.extend([shape_tensor_name])
                    op_transpose.output.extend([C])
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(C_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)

        def _split_4dim_LhsIsConstMatmul(op):
            for op_ in self._SGSModel.op:
                if op_ == op:
                    op_name = op_.name
                    A = op_.input[0]              # left matrix name
                    B = op_.input[1]              # right  matrix name
                    C = op_.output[0]             # Matrix multiply results name from A * B
                    A_tensor = self.find_tensor_by_name(A)
                    B_tensor = self.find_tensor_by_name(B)
                    C_tensor = self.find_tensor_by_name(C)
                    A_shape = A_tensor.dims
                    B_shape = B_tensor.dims
                    C_shape = self.find_tensor_by_name(C).dims # list
                    A_dim_num = len(A_tensor.dims)
                    B_dim_num = len(B_tensor.dims)
                    # OutputNeedTrans = False
                    # if self.find_tensor_by_name(A).data_format == mace_pb2.DT_NCHW or self.find_tensor_by_name(B).data_format == mace_pb2.DT_NCHW:
                    #     OutputNeedTrans = True
                    # transpose tflite input back to onnx
                    lhsChanged = False
                    rhsChanged = False
                    # A
                    # 1. create new input[1]
                    op_.input[1] = A
                    # B
                    # 2. create new input[0]
                    if B_dim_num == 4:
                        if self.is_const_tensor(B) == False:
                            op_transpose = self._SGSModel.op.add()
                            op_transpose.name = op_name + '_transposeB'
                            op_transpose.type = 'TRANSPOSE'
                            shape_tensor_name = op_transpose.name + '_shapeB'
                            shape_tensor_data = [0,1,3,2]
                            shape_tensor_shape = [4]
                            self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape,
                                            mace_pb2.DT_INT32, shape_tensor_data)
                            op_transpose.input.extend([B])
                            op_transpose.input.extend([shape_tensor_name])
                            output_op_transpose = op_transpose.name + '_outputB'
                            op_transpose.output.extend([output_op_transpose])
                            tmp_dim = [0,1,3,2]
                            tmp_shape = [0,1,2,3]
                            for i in six.moves.range(len(tmp_dim)):
                                tmp_shape[i] = B_shape[tmp_dim[i]]
                            op_transpose.output_shape.add()
                            op_transpose_shape = copy.deepcopy(tmp_shape)
                            op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                            self.add_tensor(self._SGSModel, output_op_transpose, op_transpose_shape,
                                            mace_pb2.DT_FLOAT, None)
                            self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                            B_shape = tmp_shape
                            op_.input[0] = output_op_transpose

                    # create BatchMatMul
                    op_MatMul= self._SGSModel.op.add()
                    op_MatMul.name = op_name + '_BatchMatMul'
                    op_MatMul.type = "BATCH_MATMUL"
                    op_MatMul.input.extend([op_.input[0]])
                    op_MatMul.input.extend([op_.input[1]])
                    output_op_MatMul = op_MatMul.name + '_output'
                    op_MatMul.output.extend([output_op_MatMul])
                    op_MatMul.output_shape.add()
                    output_shape = [C_shape[0],C_shape[1],C_shape[3],C_shape[2]]
                    op_MatMul.output_shape[0].dims.extend(output_shape)
                    self.add_tensor(self._SGSModel, output_op_MatMul, output_shape,
                                    mace_pb2.DT_FLOAT, None)
                    self._maceOpArray = np.append(self._maceOpArray,op_MatMul)

                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = op_name + '_transposeOut'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shapeOut'
                    shape_tensor_data = [0,1,3,2] # output won't be changed by dataformat
                    tmp_dim = [0,1,3,2]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([output_op_MatMul])
                    op_transpose.input.extend([shape_tensor_name])
                    # transpose output
                    tmp_shape = [0,1,2,3]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = C_shape[tmp_dim[i]]
                    op_transpose.output.extend([C])
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(tmp_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    self.remove_op_with_name(op_name)


        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                A = op_.input[0]              # left matrix name
                B = op_.input[1]              # right  matrix name
                A_tensor = self.find_tensor_by_name(A)
                A_dim_num = len(A_tensor.dims)
                if self.is_const_tensor(A) == True:
                    mace_check(self.is_const_tensor(B) == False, "If the lhs of MatMul is const, the rhs must be variable!" )
                    # 2dim Matmul
                    if A_dim_num == 2:
                        _split_2dim_LhsIsConstMatmul(op_)
                        self.remove_op_with_name(op_name)
                    # 3dm MatMul
                    elif A_dim_num == 3:
                        _split_3dim_LhsIsConstMatmul(op_)
                        self.remove_op_with_name(op_name)
                    # 4dim MatMul
                    elif A_dim_num == 4:
                        _split_4dim_LhsIsConstMatmul(op_)

    def split_Pad(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                pads_name = op_.input[1]
                pads = self.find_tensor_by_name(op_.input[1])
                input_tensor = self.find_tensor_by_name(op_.input[0])
                pads_list = []
                pads_back = len(pads.int32_data) // 2
                for i, j in zip(pads.int32_data[:pads_back], pads.int32_data[pads_back:]):
                    pads_list.append([i, j])
                if pads_back == 4:
                    tflite_c = pads_list[1]
                    del pads_list[1]
                    pads_list.append(tflite_c)
                pads.Clear()
                pads.name = pads_name
                pads.dims.extend([pads_back, 2])
                pads.data_type = mace_pb2.DT_INT32
                for item in pads_list:
                    pads.int32_data.extend(item)
                for arg in op_.arg:
                    if arg.name == 'pads_mode':
                        if arg.str == 'constant':
                            op_.type = "PAD"
                        elif arg.str == 'reflect':
                            op_.type = 'MIRROR_PAD'
                        else:
                            mace_check(False, 'Pad only support `constant` and `reflect` mode!')
                self._maceOpArray = np.append(self._maceOpArray, op_)
                self.OpOutputInsertTranspose(op_)


    def split_Pooling(self, op):
        # Transposed
        kernel_h,kernel_w,pad,stride = -1,-1,0,0
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                paddingL,paddingR,paddingT,paddingB = 0,0,0,0
                count_include_pad_value = 0
                xi = op_.input[0]
                op_name = op_.name
                arg = op_.arg
                inputTensor = self.find_tensor_by_name(xi)
                inputShape = self.get_shape_by_name(xi)
                if len(inputShape) == 5:
                    return self.split_Pooling3D(op_)
                elif len(inputShape) == 3:
                    return self.split_Pooling1D(op_)
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                        paddingL,paddingR,paddingT,paddingB = arg[i].ints
                    elif name == 'count_include_pad':
                        count_include_pad_value = arg[i].i
                    elif name == MaceKeyword.mace_pooling_type_str:
                        pooling_type = arg[i].i
                        if pooling_type == 1:
                            op_.type = 'AVERAGE_POOL_2D'
                        elif pooling_type == 2:
                            op_.type = 'MAX_POOL_2D'
                        elif pooling_type == 3 or pooling_type == 4: #GlobalAveragePool,GlobalMaxPool=3,4
                            input_op_name = op_.input[0]
                            input_op_shape = self.get_shape_by_name(input_op_name)
                            kernel_h = input_op_shape[1]
                            kernel_w = input_op_shape[2]
                            if pooling_type == 3:
                                op_.type = 'AVERAGE_POOL_2D'
                            elif pooling_type == 4:
                                op_.type = 'MAX_POOL_2D'
                            for i in six.moves.range(len(arg)):
                                name = arg[i].name
                                if name == MaceKeyword.mace_kernel_str:
                                    arg[i].ints[:] = []
                                    arg[i].ints.extend([kernel_h,kernel_w])
                #set padding_type
                if op_.type == 'AVERAGE_POOL_2D':
                    if count_include_pad_value == 0 and paddingL != 0:
                        padding_type = 'ONNXOUTSIDE'
                    else:
                        padding_type = 'ONNXINSIDE'
                elif op_.type == 'MAX_POOL_2D':
                    padding_type = 'CAFFE'
                else:
                    mace_check(0, "Wrong Pooling Type. Please check operator type.")
                paddingType_arg = op_.arg.add()
                paddingType_arg.name = MaceKeyword.mace_padding_str
                paddingType_arg.i = self.pooling_paddingType[padding_type]
                self._maceOpArray = np.append(self._maceOpArray,op_)
                self.OpOutputInsertTranspose(op_)


    def split_Pooling3D(self, op):
        kernel_d,kernel_h,kernel_w,pad,stride = -1,-1,-1,0,0
        for op_ in self._SGSModel.op:
            if op_ == op:
                xi = op_.input[0]
                op_name = op_.name
                inputTensor = self.find_tensor_by_name(xi)
                inputShape = self.get_shape_by_name(xi)
                output_shape = op.output_shape[0].dims
                arg = op_.arg
                strideD,strideH,strideW = 1,1,1
                dilationD,dilationH,dilationW = 1,1,1

                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                        paddingL,paddingT,paddingI,paddingR,paddingB,paddingO = arg[i].ints
                    elif name == MaceKeyword.mace_strides_str:
                        strideD,strideH,strideW = arg[i].ints
                    elif name == MaceKeyword.mace_kernel_str:
                        kernel_d,kernel_h,kernel_w = arg[i].ints

                # create transpose1->input transpose
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '_transpose1'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape1'
                shape_tensor_data = [0,2,3,4,1]
                shape_tensor_shape = [5]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([xi])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output1'
                op_transpose.output.extend([output_op_transpose])
                tmp_dim = [0,2,3,4,1]
                tmp_shape = [0,1,2,3,4]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = inputShape[tmp_dim[i]]
                op_transpose.output_shape.add()
                op_transpose_shape = copy.deepcopy(tmp_shape)
                op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                # create pooling3D
                op_pool3d= self._SGSModel.op.add()
                for i in six.moves.range(len(arg)):
                    arg_name = arg[i].name
                    if arg_name == MaceKeyword.mace_pooling_type_str:
                        pooling_type = arg[i].i
                        if pooling_type == 1:
                            op_pool3d.name = 'AvePool3D' + op_name
                        elif pooling_type == 2:
                            op_pool3d.name = 'MaxPool3D' + op_name
                        elif pooling_type == 3 or pooling_type == 4:
                            input_op_name = op_.input[0]
                            input_op_shape = self.get_shape_by_name(input_op_name)
                            kernel_d,kernel_h,kernel_w = input_op_shape[2:]
                            if pooling_type == 3:
                                op_pool3d.name = 'AvePool3D' + op_name
                            elif pooling_type == 4:
                                op_pool3d.name = 'MaxPool3D' + op_name
                            for i in six.moves.range(len(arg)):
                                arg_name = arg[i].name
                                if arg_name == MaceKeyword.mace_kernel_str:
                                    arg[i].ints[:] = []
                                    arg[i].ints.extend([kernel_d,kernel_h,kernel_w])
                op_pool3d.type = 'CUSTOM'

                #create arg
                strides_arg = op_pool3d.arg.add()
                strides_arg.name = 'strides'
                strides_arg.ints.extend([strideD,strideH,strideW])
                padding_arg = op_pool3d.arg.add()
                padding_arg.name = 'padding_values'
                padding_arg.ints.extend([paddingL,paddingR,paddingT,paddingB,paddingI,paddingO])
                kernels_arg = op_pool3d.arg.add()
                kernels_arg.name = 'kernels'
                kernels_arg.ints.extend([kernel_d,kernel_h,kernel_w])

                # create input X
                op_pool3d.input.extend([output_op_transpose])

                #create output
                output_op_pool3d = op_pool3d.name + '_output'
                op_pool3d.output.extend([output_op_pool3d])
                op_pool3d.output_shape.add()
                tmp_dim = [0,2,3,4,1]
                tmp_shape = [0,1,2,3,4]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = output_shape[tmp_dim[i]]
                pool3d_output_shape = tmp_shape
                op_pool3d.output_shape[0].dims.extend(pool3d_output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_pool3d)

                # creat transpose from NCHW back to NHWC
                op_transpose3 = self._SGSModel.op.add()
                op_transpose3.name = op_name + '_transpose3'
                op_transpose3.type = 'TRANSPOSE'
                shape_tensor_name3 = op_transpose3.name + '_shape3'
                shape_tensor_data3 = [0,4,1,2,3]
                shape_tensor_shape = [5]
                self.add_tensor(self._SGSModel ,shape_tensor_name3, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data3)
                op_transpose3.input.extend([output_op_pool3d])
                op_transpose3.input.extend([shape_tensor_name3])
                op_transpose3.output.extend(op.output)
                op_transpose3.output_shape.add()
                op_transpose3.output_shape[0].dims.extend(output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose3)

    def split_Pooling1D(self, op):
        kernel_h,kernel_w,pad,stride = -1,-1,0,0
        for op_ in self._SGSModel.op:
            if op_ == op:
                paddingL,paddingR,paddingT,paddingB = 0,0,0,0
                count_include_pad_value = 0
                xi = op_.input[0]
                op_name = op_.name
                arg = op_.arg
                inputTensor = self.find_tensor_by_name(xi)
                inputShape = self.get_shape_by_name(xi)
                [n,c,w] = op_.output_shape[0].dims[:]

                # create  transpose op
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '#transpose'
                op_transpose.type = 'TRANSPOSE'
                op_transpose.input.extend([xi])
                # set transpose's output_shape
                transpose_output_tensor_name = op_transpose.name + '_output_shape'
                transpose_output_tensor_data = [0, 2, 1]  # n,w,c
                transpose_output_tensor_shape = [3]
                self.add_tensor(self._SGSModel, transpose_output_tensor_name, transpose_output_tensor_shape,
                                mace_pb2.DT_INT32, transpose_output_tensor_data)
                op_transpose.input.extend([transpose_output_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend([inputShape[0], inputShape[2], inputShape[1]])
                self._maceOpArray = np.append(self._maceOpArray, op_transpose)

                # create reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '#reshape1'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_transpose])
                # set reshape's output_shape
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [op_transpose.output_shape[0].dims[0], 1, op_transpose.output_shape[0].dims[1],
                                              op_transpose.output_shape[0].dims[2]]  # n,h,w,c
                reshape_output_tensor_shape = [4]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend([op_transpose.output_shape[0].dims[0], 1, op_transpose.output_shape[0].dims[1],
                                              op_transpose.output_shape[0].dims[2]])  # nchw
                self._maceOpArray = np.append(self._maceOpArray, op_reshape)

                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                    #padding value should be divided by 2
                        paddingL,paddingR,paddingT,paddingB = arg[i].ints
                    elif name == MaceKeyword.mace_strides_str:
                        strideH,strideW = arg[i].ints
                    elif name == MaceKeyword.mace_kernel_str:
                        kernel_h,kernel_w = arg[i].ints[:]
                    elif name == MaceKeyword.mace_pooling_type_str:
                        pooling_type = arg[i].i
                        if pooling_type == 1:
                            op_.type = type_str = 'AVERAGE_POOL_2D'
                        elif pooling_type == 2:
                            op_.type = type_str = 'MAX_POOL_2D'
                        elif pooling_type == 3 or pooling_type == 4: #GlobalAveragePool,GlobalMaxPool=3,4
                            kernel_h,kernel_w = [1, op_transpose.output_shape[0].dims[1]]
                            if pooling_type == 3:
                                op_.type = type_str = 'AVERAGE_POOL_2D'
                            elif pooling_type == 4:
                                op_.type = type_str = 'MAX_POOL_2D'

                # ADD pool
                op_pool = self._SGSModel.op.add()
                op_pool.name = op_name + type_str
                op_pool.type = op_.type
                #add padding_type
                paddingType_arg = op_pool.arg.add()
                paddingType_arg.name = MaceKeyword.mace_padding_str
                #set padding_type
                paddingType_arg.i = self.pooling_paddingType['CAFFE']
                strides_arg = op_pool.arg.add()
                strides_arg.name = 'strides'
                strides_arg.ints.extend([strideH, strideW])
                padding_arg = op_pool.arg.add()
                padding_arg.name = 'padding_values'
                padding_arg.ints.extend([paddingL, paddingR, paddingT, paddingB])
                kernel_arg = op_pool.arg.add()
                kernel_arg.name = 'kernels'
                kernel_arg.ints.extend([kernel_h,kernel_w])
                op_pool.input.extend([output_op_reshape])
                output_op_pool = op_pool.name + '_output'
                op_pool.output.extend([output_op_pool])
                op_pool.output_shape.add()
                op_pool.output_shape[0].dims.extend([n,1,w,c])  # nchw
                self._maceOpArray = np.append(self._maceOpArray, op_pool)
                # output_shape = [1,a,b,c]
                # output_op_pool = op_name + '_pool_output'
                # self.create_Pooling_op(op_, op_.type, output_op_pool, output_shape)

                # create reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '#reshape2'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_pool])
                # set reshape's output_shape
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [n,w,c]  # n,h,w,c
                reshape_output_tensor_shape = [3]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend([n,w,c])
                self._maceOpArray = np.append(self._maceOpArray, op_reshape)

                # create  transpose op
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '#transpose2'
                op_transpose.type = 'TRANSPOSE'
                op_transpose.input.extend([output_op_reshape])
                # set transpose's output_shape
                transpose_output_tensor_name = op_transpose.name + '_output_shape'
                transpose_output_tensor_data = [0, 2, 1]  # n,c,w,h
                transpose_output_tensor_shape = [3]
                self.add_tensor(self._SGSModel, transpose_output_tensor_name, transpose_output_tensor_shape,
                                mace_pb2.DT_INT32, transpose_output_tensor_data)
                op_transpose.input.extend([transpose_output_tensor_name])
                op_transpose.output[:] = op.output[:]
                #output_op_transpose2 = op_transpose2.name + '_output'
                #op_transpose2.output.extend([output_op_transpose2])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend([n,c,w])
                self._maceOpArray = np.append(self._maceOpArray, op_transpose)

                #set padding_type
                if op_.type == 'AVERAGE_POOL_2D':
                    if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
                        padding_type = 'ONNXOUTSIDE'
                    else:
                        if count_include_pad_value == 0 and paddingL != 0:
                            padding_type = 'ONNXOUTSIDE'
                        else:
                            padding_type = 'ONNXINSIDE'
                elif op_.type == 'MAX_POOL_2D':
                    padding_type = 'CAFFE'
                else:
                    mace_check(0, "Wrong Pooling Type. Please check operator type.")
                paddingType_arg = op_.arg.add()
                paddingType_arg.name = MaceKeyword.mace_padding_str
                paddingType_arg.i = self.pooling_paddingType[padding_type]
                self.remove_op_with_name(op_name)


    def split_PriorBox(self, op):
        # unTransposed
        #init param
        max_sizes = []
        step_h,step_w,img_h,img_w =0,0,0,0
        op_name = op.name
        input_conv_name = op.input[0]
        data = op.input[1]
        output_name = op.output[0]

        for op_ in self._oriModel.op:
          # find input conv shape
          for outputName in op_.output:
            if input_conv_name == outputName:
              [layer_height,layer_width] = op_.output_shape[0].dims[2:]
          #find para
          if op_ == op:
            output_shape = op_.output_shape[0].dims
            arg = op_.arg
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == 'min_size':
                min_sizes = arg[i].floats
              elif name == 'max_size':
                max_sizes = arg[i].floats
              elif name == 'flip':
                flip = arg[i].i
              elif name == 'aspect_ratio':
                aspect_ratios = arg[i].floats
              elif name == 'clip':
                clip = arg[i].i
              elif name == 'variance':
                variance = arg[i].floats
              elif name == 'offset':
                offset = arg[i].f
              elif name == 'step_h':
                step_h = arg[i].f
              elif name == 'step_w':
                step_w = arg[i].f
        [img_height,img_width] = self._inputShape[0][2:]
        num_prior = len(aspect_ratios) * len(min_sizes) + len(max_sizes)

        #gen anchor box
        top_data = SGSModel_genAnchors.genAnchors(layer_width,layer_height,img_width,img_height,
                                                  aspect_ratios,variance,offset,num_prior,min_sizes,
                                                  max_sizes,clip,step_h,step_w,img_h,img_w)
        #remove origin priorBox out tensor
        self.remove_tensor_by_name(output_name)

        #add tensor
        self.add_tensor(self._SGSModel ,output_name, output_shape,
                        mace_pb2.DT_FLOAT, top_data)
        #remove priorBox op
        self.remove_op_with_name(op_name)

    def split_PriorBox_RFC(self, op):
        # unTransposed
            #init param
            max_sizes = []
            step_h,step_w,img_h,img_w =0,0,0,0
            op_name = op.name
            input_conv_name = op.input[0]
            data = op.input[1]
            output_name = op.output[0]

            for op_ in self._oriModel.op:
              # find input conv shape
              for outputName in op_.output:
                if input_conv_name == outputName:
                  [layer_height,layer_width] = op_.output_shape[0].dims[2:]
              #find para
              if op_ == op:
                output_shape = op_.output_shape[0].dims
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                  name = arg[i].name
                  if name == 'min_size':
                    min_sizes = arg[i].floats
                  elif name == 'max_size':
                    max_sizes = arg[i].floats
                  elif name == 'flip':
                    flip = arg[i].i
                  elif name == 'aspect_ratio':
                    aspect_ratios = arg[i].floats
                  elif name == 'clip':
                    clip = arg[i].i
                  elif name == 'variance':
                    variance = arg[i].floats
                  elif name == 'offset':
                    offset = arg[i].f
                  elif name == 'step_h':
                    step_h = arg[i].f
                  elif name == 'step_w':
                    step_w = arg[i].f

            [img_height,img_width] = self._inputShape[0][2:]
            num_prior = len(aspect_ratios) * len(min_sizes) + len(max_sizes)
            #gen anchor box
            top_data = SGSModel_prior_box.genAnchors(layer_width,layer_height,img_width,img_height,
                                                      aspect_ratios,variance,offset,num_prior,min_sizes,
                                                      max_sizes,clip,step_h,step_w,img_h,img_w)
            #remove origin priorBox out tensor
            self.remove_tensor_by_name(output_name)

            #add tensor
            self.add_tensor(self._SGSModel ,output_name, output_shape,
                            mace_pb2.DT_FLOAT, top_data)
            #remove priorBox op
            self.remove_op_with_name(op_name)

    def split_Reduce(self, op):
        # Transposed according keepdim
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                arg = op_.arg
                reduce_type = "Unknow"
                axis = -1
                axis_ori = 0
                axis_data = []
                axis_exist = False
                input_shape = self.get_shape_by_name(op_.input[0])
                # find op type
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_reduce_type_str:
                        type = arg[i].i
                        # MEAN = 0 MIN = 1 MAX = 2 PROD = 3 SUM = 4
                        if type == 0:
                            reduce_type = "MEAN"
                            return self.split_Reduce_Mean(op_)
                        elif type == 1:
                            reduce_type = 'REDUCE_MIN'
                        elif type == 2:
                            reduce_type = 'REDUCE_MAX'
                        elif type == 4:
                            reduce_type = 'SUM'
                        else:
                            mace_check(False,'Reduce do not support.')
                    elif name == MaceKeyword.mace_keepdims_str: # "keepdims"
                        keepdim = arg[i].i

                if keepdim:
                    self.OpInputInsertTranspose(op_)
                    input_shape = self.get_shape_by_name(op_.input[0])
                    inputTensor = self.find_tensor_by_name(op_.input[0])
                    output_shape = op_.output_shape[0].dims[:]
                    input = op_.input[0]
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str: # "axis"
                            for j in range(len(arg[i].ints)):
                                axis_exist = True
                                axis_ori = self.get_axes_by_shape (arg[i].ints[j], len(input_shape), True, True)
                                axis_new = axis_ori
                                axis_data.append(axis_new)
                                axis_data.sort()
                    # if axis = None
                    if axis_exist == False:
                        axis_data = [x for x in range(len(input_shape))]
                        ori_axis_data = [x for x in range(len(input_shape))]
                    # creat axis tensor
                    axis_tensor_name = op_name + '_axis'
                    axis_tensor_data = axis_data
                    axis_tensor_shape = [len(axis_data)]
                    self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                                    mace_pb2.DT_INT32, axis_tensor_data)

                    op_.type = reduce_type
                    op_.input.extend([axis_tensor_name])
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)
                else:
                    input_shape = self.get_shape_by_name(op_.input[0])
                    inputTensor = self.find_tensor_by_name(op_.input[0])
                    output_shape = op_.output_shape[0].dims[:]
                    input = op_.input[0]
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str: # "axis"
                            for j in range(len(arg[i].ints)):
                                axis_exist = True
                                axis_ori = self.get_axes_by_shape (arg[i].ints[j], len(input_shape), False, True)
                                axis_new = axis_ori
                                axis_data.append(axis_new)
                                axis_data.sort()
                    # if axis = None
                    if axis_exist == False:
                        axis_data = [x for x in range(len(input_shape))]
                        ori_axis_data = [x for x in range(len(input_shape))]
                    # creat axis tensor
                    axis_tensor_name = op_name + '_axis'
                    axis_tensor_data = axis_data
                    axis_tensor_shape = [len(axis_data)]
                    self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                                    mace_pb2.DT_INT32, axis_tensor_data)

                    op_.type = reduce_type
                    op_.input.extend([axis_tensor_name])
                    self._maceOpArray = np.append(self._maceOpArray,op_)


    def split_Clip(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_name = op_.name
                output_shape = op.output_shape[0].dims[:]
                if len(output_shape) == 4:
                    output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                xi = op_.input[0]
                arg = op_.arg
                value_list = []
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'coeff':
                        value_list = arg[i].floats
                minTensor_data = value_list[0]
                maxTensor_data = value_list[1]

                op_clip = self._SGSModel.op.add()
                op_clip.name = 'clip' + op_name
                op_clip.type = 'CUSTOM'
                min_tensor_name = op_clip.name + '_min'
                min_tensor_data = [minTensor_data]
                min_tensor_shape = [1]
                self.add_tensor(self._SGSModel ,min_tensor_name, min_tensor_shape,
                    mace_pb2.DT_FLOAT, min_tensor_data)
                max_tensor_name = op_clip.name + '_max'
                max_tensor_data = [maxTensor_data]
                max_tensor_shape = [1]
                self.add_tensor(self._SGSModel ,max_tensor_name, max_tensor_shape,
                    mace_pb2.DT_FLOAT, max_tensor_data)
                op_clip.input.extend([xi])
                op_clip.input.extend([min_tensor_name])
                op_clip.input.extend([max_tensor_name])
                op_clip.output[:] = op.output[:]
                op_clip.output_shape.add()
                op_clip.output_shape[0].dims.extend(output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_clip)
                self.OpOutputInsertTranspose(op_clip)
                self.remove_op_with_name(op_name)

    def split_ReduceL2(self,op):
        # unTransposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                xi = op_.input[0]
                inputTensor = self.find_tensor_by_name(xi)
                output = op_.output[0]
                outputTensor = self.find_tensor_by_name(output)
                input_shape = self.get_shape_by_name(xi)
                output_shape = op.output_shape[0].dims[:]
                arg = op_.arg
                power = 2
                axis_ori = 0
                axis_data = []
                ori_axis_data = []
                need_transpose = False
                keepdim = 1

                # get attrs
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_axis_str: # "axis"
                        for j in range(len(arg[i].ints)):
                            axis_ori = len(input_shape) + arg[i].ints[j] if arg[i].ints[j] < 0 else arg[i].ints[j]
                            axis_new = axis_ori
                            ori_axis_data.append(axis_ori)
                            ori_axis_data.sort()
                            axis_data.append(axis_new)
                            axis_data.sort()
                    if name == MaceKeyword.mace_keepdims_str: # "keepdims"
                        keepdim = arg[i].i

                # creat axis tensor
                axis_tensor_name = op_name + '_axis'
                axis_tensor_data = axis_data
                axis_tensor_shape = [len(axis_data)]
                self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                                mace_pb2.DT_INT32, axis_tensor_data)

                # creat square
                loop_num = int(power)-1 # 1
                for i in six.moves.range(loop_num):
                    op_mul = self._SGSModel.op.add()
                    op_mul.name = op_name + '_MUL#' + str(i)
                    op_mul.type = 'MUL'
                    op_mul.input.extend([xi])
                    op_mul.input.extend([xi])
                    output_name_mul = op_mul.name + '_output'
                    op_mul.output.extend([output_name_mul])
                    #op_mul.output_shape.extend(op.input_shape)
                    op_mul.output_shape.add()
                    op_mul.output_shape[0].dims.extend(input_shape)
                    self.add_tensor(self._SGSModel, output_name_mul, input_shape,
                                    mace_pb2.DT_FLOAT, None)
                    #tensor data is variable,so didn't creat new tensor
                    self._maceOpArray = np.append(self._maceOpArray,op_mul)

                # creat reduce_sum
                op_reduceSum = self._SGSModel.op.add()
                op_reduceSum.name = op_name + '_reduceSum'
                op_reduceSum.type = 'SUM'
                keepdims_arg = op_reduceSum.arg.add()
                keepdims_arg.name = 'keepdims'
                keepdims_arg.i = keepdim
                op_reduceSum.input.extend([output_name_mul])
                op_reduceSum.input.extend([axis_tensor_name])
                output_op_reduceSum = op_reduceSum.name + '_output'
                op_reduceSum.output.extend([output_op_reduceSum])
                op_reduceSum.output_shape.add()
                op_reduceSum.output_shape[0].dims.extend(output_shape)
                self.add_tensor(self._SGSModel, output_op_reduceSum, output_shape,
                                mace_pb2.DT_FLOAT, None)
                self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)

                # creat sqrt
                op_sqrt = self._SGSModel.op.add()
                op_sqrt.name = op_name + '_sqrt'
                op_sqrt.type = 'SQRT'
                op_sqrt.input.extend([output_op_reduceSum])
                op_sqrt.output[:] =  op.output[:]
                op_sqrt.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_sqrt)
                self.remove_op_with_name(op_name)

    def split_Reduce_Mean(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                input_shape_ori = self.get_shape_by_name(op_.input[0])
                output_shape_ori = op_.output_shape[0].dims[:]
                if len(input_shape_ori) == 4 and len(output_shape_ori) == 4:
                    self.OpInputInsertTranspose(op_)
                op_name = op_.name
                arg = op_.arg
                axis = -1
                axis_ori = 0
                axis_data = []
                xi = op_.input[0]
                input_shape = self.get_shape_by_name(op_.input[0])
                output_shape = op_.output_shape[0].dims[:]
                axis_exist = False
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_axis_str:
                        for j in range(len(arg[i].ints)):
                            axis_exist = True
                            axis_ori = len(input_shape) + arg[i].ints[j] if arg[i].ints[j] < 0 else arg[i].ints[j]
                            if len(input_shape) == 4 and len(output_shape) == 4:
                                if axis_ori == 0:
                                    axis_new = 0
                                elif axis_ori == 2:
                                    axis_new = 1
                                elif axis_ori == 1:
                                    axis_new = 3
                                elif axis_ori == 3:
                                    axis_new =2
                            else:
                                axis_new = axis_ori
                            axis_data.append(axis_new)
                            axis_data.sort()
                    if name == MaceKeyword.mace_keepdims_str: # "keepdims"
                        keepdim = arg[i].i

                # if axis = None
                if axis_exist == False:
                    axis_data = [x for x in range(len(input_shape))]
                    ori_axis_data = [x for x in range(len(input_shape))]

                # creat axis tensor
                axis_tensor_name = op_name + '_axis'
                axis_tensor_data = axis_data
                axis_tensor_shape = [len(axis_data)]
                self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                mace_pb2.DT_INT32, axis_tensor_data)

                # deal 4dimension case
                if len(input_shape) == 4 and len(output_shape) == 4:
                    input = op_.input[0]
                    if axis_data == [1,2]:
                        op_.type = "MEAN"
                        op_.input.pop()
                        op_.input.extend([input])
                        op_.input.extend([axis_tensor_name])
                        self._maceOpArray = np.append(self._maceOpArray,op_)
                        self.OpOutputInsertTranspose(op_)
                    else:
                        input = op_.input[0]
                        reduce_dim = input_shape[axis_new]
                        #creat reduceSum
                        op_reduceSum = self._SGSModel.op.add()
                        op_reduceSum.name = op_name + '_reduceSum'
                        op_reduceSum.type = 'SUM'
                        keepdims_arg = op_reduceSum.arg.add()
                        keepdims_arg.name = 'keepdims'
                        keepdims_arg.i = keepdim
                        op_reduceSum.input.extend([input])
                        op_reduceSum.input.extend([axis_tensor_name])
                        output_op_reduceSum = op_reduceSum.name + '_output'
                        op_reduceSum.output.extend([output_op_reduceSum])
                        op_reduceSum.output_shape.add()
                        if len(output_shape) == 4:
                            output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                        op_reduceSum.output_shape[0].dims.extend(output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)
                        #creat div
                        for i in range(len(axis_data)):
                            axis_new = axis_data[i]
                            if i == 0:
                                reduce_dim = input_shape[axis_new]
                            else:
                                reduce_dim *= input_shape[axis_new]
                        const_tensor_name = op_name + '_const'
                        const_tensor_data = [reduce_dim]
                        const_tensor_shape = [1]
                        self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                        mace_pb2.DT_FLOAT, const_tensor_data)
                        op_div = self._SGSModel.op.add()
                        op_div.name = op_name + '_div'
                        op_div.type = 'DIV'
                        op_div.input.extend([output_op_reduceSum])
                        op_div.input.extend([const_tensor_name])
                        op_div.output[:] =  op_.output[:]
                        output_shape = op_.output_shape[0].dims[:]
                        op_div.output_shape.add()
                        if len(output_shape) == 4:
                            op_div.output_shape[0].dims.extend([output_shape[0],output_shape[2],output_shape[3],output_shape[1]])
                        else:
                            op_div.output_shape[0].dims.extend(output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_div)
                        self.OpOutputInsertTranspose(op_div)
                        self.remove_op_with_name(op_name)
                else:
                    reduce_dim = input_shape[axis_new]
                    #creat reduceSum
                    op_reduceSum = self._SGSModel.op.add()
                    op_reduceSum.name = op_name + '_reduceSum'
                    op_reduceSum.type = 'SUM'
                    keepdims_arg = op_reduceSum.arg.add()
                    keepdims_arg.name = 'keepdims'
                    keepdims_arg.i = keepdim
                    op_reduceSum.input.extend([xi])
                    op_reduceSum.input.extend([axis_tensor_name])
                    output_op_reduceSum = op_reduceSum.name + '_output'
                    op_reduceSum.output.extend([output_op_reduceSum])
                    op_reduceSum.output_shape.add()
                    if len(output_shape) == 4:
                        output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                    op_reduceSum.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)
                    #creat div
                    for i in range(len(axis_data)):
                        axis_new = axis_data[i]
                        if i == 0:
                            reduce_dim = input_shape[axis_new]
                        else:
                            reduce_dim *= input_shape[axis_new]
                    const_tensor_name = op_name + '_const'
                    const_tensor_data = [reduce_dim]
                    const_tensor_shape = [1]
                    self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                    mace_pb2.DT_FLOAT, const_tensor_data)
                    op_div = self._SGSModel.op.add()
                    op_div.name = op_name + '_div'
                    op_div.type = 'DIV'
                    op_div.input.extend([output_op_reduceSum])
                    op_div.input.extend([const_tensor_name])
                    op_div.output[:] =  op_.output[:]
                    output_shape = op_.output_shape[0].dims[:]
                    op_div.output_shape.add()
                    if len(output_shape) == 4:
                        op_div.output_shape[0].dims.extend([output_shape[0],output_shape[2],output_shape[3],output_shape[1]])
                    else:
                        op_div.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_div)
                    self.remove_op_with_name(op_name)

    def split_Reshape(self, op):
        # unTransposed
        xi = op.input[0]
        op_name = op.name
        inputTensor = self.find_tensor_by_name(xi)
        input_shape = self.get_shape_by_name(xi)
        output_shape = self.get_shape_by_name(op.output[0])
        input_shape_total = 1
        output_shape_total = 1
        for shape in input_shape:
            input_shape_total *= shape
        for shape in output_shape:
            output_shape_total *= shape
        mace_check(input_shape_total == output_shape_total,
            "reshape op '%s' input/output tensor must have same total shape value\n"
            % op_name)

        for op_ in self._SGSModel.op:
            if op_ == op:
                #output_shape = op_.output_shape[0].dims[:]
                output_tensor_name = op_.name + '_output_shape'
                output_tensor_data = output_shape
                output_tensor_shape = [len(output_shape)]
                self.add_tensor(self._SGSModel, output_tensor_name, output_tensor_shape,
                                mace_pb2.DT_INT32, output_tensor_data)
                op_.input.pop()
                op_.input.extend([output_tensor_name])
                op_.type = "RESHAPE"
                self._maceOpArray = np.append(self._maceOpArray,op_)


    def split_Slice(self, op):
        # unTransposed
        for op_ in self._SGSModel.op:
          if op_ == op:
            xi = op_.input[0]
            input_tensor = self.find_tensor_by_name(xi)
            input_shape = self.get_shape_by_name(xi)
            input_number = len(op_.input)
            arg = op_.arg
            arg_index = 0

            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == op_.name + '_starts':
                starts_arg = arg[i].ints[:]
              elif name == op_.name + '_ends':
                ends_arg = arg[i].ints[:]
              elif name == op.name + '_axis':
                axes_arg = arg[i].ints[:]
              elif name == op.name + '_steps':
                steps_arg = arg[i].ints[:]

            begin_tensor = list(np.zeros((len(input_shape)), dtype=np.int))   # default all 0
            end_tensor = copy.deepcopy(input_tensor.dims)                     # default all max
            axis_tensor = [False for i in six.moves.range(len(input_shape))]  # default 0 to dim
            strides_tensor = list(np.ones((len(input_shape)), dtype=np.int))  # default all 1

            for axis in axes_arg:
              axis = len(input_shape) + axis if axis < 0 else axis
              axis_tensor[axis] = True

            # generate begin/end/stride tensor according to axis
            for i in six.moves.range(len(input_shape)):
              if axis_tensor[i] == True:
                begin_tensor[i] = starts_arg[arg_index]
                if begin_tensor[i] < 0:
                  begin_tensor[i] = begin_tensor[i] + input_tensor.dims[i]
                elif begin_tensor[i] > input_tensor.dims[i]:
                  begin_tensor[i] = input_tensor.dims[i]
                end_tensor[i] = ends_arg[arg_index]
                if end_tensor[i] < 0:
                  end_tensor[i] = end_tensor[i] + input_tensor.dims[i]
                elif end_tensor[i] > input_tensor.dims[i]:
                  end_tensor[i] = input_tensor.dims[i]
                strides_tensor[i] = steps_arg[arg_index]
                arg_index = arg_index + 1

            # remove origin begin/end/axis/stride tensors
            for i in six.moves.range((input_number-1)):
              op_.input.pop()

            # create new begin/end/stride tensors
            op_.type = "STRIDED_SLICE"
            op_.input.extend([op_.name + '_begin'])
            self.add_tensor(self._SGSModel, op_.input[1], [len(begin_tensor)],
                            mace_pb2.DT_INT32, begin_tensor)
            op_.input.extend([op_.name + '_finial'])
            self.add_tensor(self._SGSModel, op_.input[2], [len(end_tensor)],
                            mace_pb2.DT_INT32, end_tensor)
            op_.input.extend([op_.name + '_strides'])
            self.add_tensor(self._SGSModel, op_.input[3], [len(strides_tensor)],
                            mace_pb2.DT_INT32, strides_tensor)
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Split(self, op):
        # Transposed by discuss
        slice_point_enable = False
        use_slice = False
        output_detached = []
        for i in six.moves.range(len(op.output)):
            if self.is_tensor_detached(op.output[i]):
                use_slice = True
                output_detached.extend([1])
            else:
                output_detached.extend([0])

        if use_slice == False:
            for op_ in self._SGSModel.op:
                if op_ == op:
                    self.OpInputInsertTranspose(op_)
                    arg = op_.arg
                    output_shape = op_.output_shape[:]
                    op_input = op_.input[0]
                    inputShape = self.get_shape_by_name (op_input)
                    inputTensor = self.find_tensor_by_name(op_input)
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            arg[i].i = self.get_axes_by_shape(arg[i].i, len(inputShape),True,True)
                            axis_tensor_name = op_.name + '_axis'
                            axis_tensor_data = [op_.arg[i].i]
                            axis_tensor_shape = [1]
                            self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                                            mace_pb2.DT_INT32, axis_tensor_data)
                        elif name == 'slice_point':
                            slice_point_enable = True
                            slice_point = arg[i].ints
                            slice_point_name = op_.name + '_slice_point'
                            slice_point_data = slice_point
                            slice_point_shape = [len(slice_point)]
                            self.add_tensor(self._SGSModel, slice_point_name, slice_point_shape,
                                            mace_pb2.DT_INT32, slice_point_data)
                    #creat split or splitV
                    if slice_point_enable: # creat split_v
                        op_.input.extend([slice_point_name])
                        op_.input.extend([axis_tensor_name])
                        op_.type = "SPLIT_V"
                    else:
                        op_.input[0] = axis_tensor_name
                        op_.input.extend([op_input])
                        op_.type = "SPLIT"
                    self._maceOpArray = np.append(self._maceOpArray, op_)
                    self.OpOutputInsertTranspose(op_)
        else:
            for op_ in self._SGSModel.op:
                if op_ == op:
                    arg = op_.arg
                    output_shape = op_.output_shape[:]
                    op_input = op_.input[0]
                    inputShape = self.get_shape_by_name (op_input)
                    inputTensor = self.find_tensor_by_name(op_input)
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            axis = self.get_axes_by_shape (arg[i].i, len(inputShape), False)
                        elif name == 'slice_point':
                            slice_point = arg[i].ints
                            slice_point_enable = True

                    if not slice_point_enable:
                        slice_size = inputShape[axis] / len(op_.output_shape)
                        slice_point = [slice_size] * len(op_.output_shape)

                    # generate begin/size tensor according to axis
                    begin_position = 0
                    for i in six.moves.range(len(op_.output_shape)):
                        begin_tensor_data = list(np.zeros((len(inputShape)), dtype=np.int))   # default all 0
                        size_tensor_data = copy.deepcopy(inputTensor.dims)                    # default all max
                        for j in six.moves.range(len(inputShape)):
                            if j == axis:
                                begin_tensor_data[j] = begin_position
                                size_tensor_data[j] = slice_point[i]
                        begin_position = begin_position + slice_point[i]

                        if output_detached[i] == 0:
                            op_slice = self._SGSModel.op.add()
                            op_slice.name = op_.name + '_slice#' + str(i)
                            op_slice.type = 'SLICE'
                            begin_tensor_name = op_slice.name + '_begin'
                            begin_tensor_shape = [len(inputShape)]
                            self.add_tensor(self._SGSModel, begin_tensor_name, begin_tensor_shape,
                                            mace_pb2.DT_INT32, begin_tensor_data)
                            size_tensor_name = op_slice.name + '_size'
                            size_tensor_shape = [len(inputShape)]
                            self.add_tensor(self._SGSModel, size_tensor_name, size_tensor_shape,
                                            mace_pb2.DT_INT32, size_tensor_data)
                            op_slice.input.extend([op_input])
                            op_slice.input.extend([begin_tensor_name])
                            op_slice.input.extend([size_tensor_name])
                            op_slice.output.extend([op_.output[i]])
                            op_slice.output_shape.add()
                            op_slice.output_shape[0].dims.extend(size_tensor_data)
                            self._maceOpArray = np.append(self._maceOpArray,op_slice)
                    self.remove_op_with_name(op_.name)

    def split_Softmax(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                xi = op.input[0]
                op_name = op.name
                slice_point_enable = False
                inputTensor = self.find_tensor_by_name(xi)
                inputShape = self.get_shape_by_name(xi)
                output_shape = op.output_shape[0].dims[:]
                arg = op.arg
                shape_tensor_data = [x for x in range(0, len(inputShape))]

                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'axis':
                        arg[i].i = len(output_shape) + arg[i].i if arg[i].i < 0 else arg[i].i
                        axis= arg[i].i
                        mace_check(axis < len(output_shape) and axis >= 0,
                                " axis must be within length of input dims\n")

                # SGS Softmax only do softmax on innermost dim, so need to transpose according to axis param

                # last axis will be keeped at innermost dim. No need for special treatment
                if len(inputShape) == 4:
                    if axis == len(inputShape) -1:
                        op_.type = "SOFTMAX"
                        self._maceOpArray = np.append(self._maceOpArray,op_)
                    elif axis == 1:
                        self.OpInputInsertTranspose(op_)
                        op_.type = "SOFTMAX"
                        self._maceOpArray = np.append(self._maceOpArray,op_)
                        self.OpOutputInsertTranspose(op_)
                    # For other cases, transpose dim pointed by arg axis
                    else:
                        temp = shape_tensor_data[axis]
                        shape_tensor_data[axis] = shape_tensor_data[-1]
                        shape_tensor_data[-1] = temp

                        # creat transpose for switch axis to innermost dim
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = 'SGS_' + op_name + '_transpose'
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shape1'
                        shape_tensor_shape = [len(inputShape)]
                        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                            mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([xi])
                        op_transpose.input.extend([shape_tensor_name])
                        output_op_transpose = op_transpose.name + '_output1'
                        op_transpose.output.extend([output_op_transpose])
                        tmp_shape = [x for x in range(0, len(inputShape))]
                        for i in six.moves.range(len(inputShape)):
                            if i == axis:
                                tmp_shape[i] = inputShape[-1]
                            elif i == len(inputShape) -1:
                                tmp_shape[i] = inputShape[axis]
                            else:
                                tmp_shape[i] = inputShape[i]

                        op_transpose.output_shape.add()
                        op_transpose.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                        # ADD SOFTMAX
                        op_softmax= self._SGSModel.op.add()
                        op_softmax.name = op_name + '_softmax'
                        op_softmax.type = 'SOFTMAX'
                        op_softmax.input.extend([output_op_transpose])
                        output_op_softmax = op_softmax.name + '_output'
                        op_softmax.output.extend([output_op_softmax])
                        op_softmax.output_shape.add()
                        op_softmax.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_softmax)

                        # creat transpose for switch axis to innermost dim to resume tensor shape
                        op_transpose2 = self._SGSModel.op.add()
                        op_transpose2.name = op_name + '_transpose2'
                        op_transpose2.type = 'TRANSPOSE'
                        shape_tensor_name2 = op_transpose2.name + '_shape2'
                        shape_tensor_shape = [len(inputShape)]
                        self.add_tensor(self._SGSModel ,shape_tensor_name2, shape_tensor_shape,
                            mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose2.input.extend([output_op_softmax])
                        op_transpose2.input.extend([shape_tensor_name2])
                        op_transpose2.output.extend(op.output)
                        op_transpose2.output_shape.add()
                        op_transpose2.output_shape[0].dims.extend(output_shape)

                        self._maceOpArray = np.append(self._maceOpArray,op_transpose2)
                        self.remove_op_with_name(op_name)
                else:
                    if axis == len(inputShape) -1:
                        op_.type = "SOFTMAX"
                        self._maceOpArray = np.append(self._maceOpArray,op_)
                    else:
                        temp = shape_tensor_data[axis]
                        shape_tensor_data[axis] = shape_tensor_data[-1]
                        shape_tensor_data[-1] = temp

                        # creat transpose for switch axis to innermost dim
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = 'SGS_' + op_name + '_transpose'
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shape1'
                        shape_tensor_shape = [len(inputShape)]
                        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                            mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([xi])
                        op_transpose.input.extend([shape_tensor_name])
                        output_op_transpose = op_transpose.name + '_output1'
                        op_transpose.output.extend([output_op_transpose])
                        tmp_shape = [x for x in range(0, len(inputShape))]
                        for i in six.moves.range(len(inputShape)):
                            if i == axis:
                                tmp_shape[i] = inputShape[-1]
                            elif i == len(inputShape) -1:
                                tmp_shape[i] = inputShape[axis]
                            else:
                                tmp_shape[i] = inputShape[i]

                        op_transpose.output_shape.add()
                        op_transpose.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                        # ADD SOFTMAX
                        op_softmax= self._SGSModel.op.add()
                        op_softmax.name = op_name + '_softmax'
                        op_softmax.type = 'SOFTMAX'
                        op_softmax.input.extend([output_op_transpose])
                        output_op_softmax = op_softmax.name + '_output'
                        op_softmax.output.extend([output_op_softmax])
                        op_softmax.output_shape.add()
                        op_softmax.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_softmax)

                        # creat transpose for switch axis to innermost dim to resume tensor shape
                        op_transpose2 = self._SGSModel.op.add()
                        op_transpose2.name = op_name + '_transpose2'
                        op_transpose2.type = 'TRANSPOSE'
                        shape_tensor_name2 = op_transpose2.name + '_shape2'
                        shape_tensor_shape = [len(inputShape)]
                        self.add_tensor(self._SGSModel ,shape_tensor_name2, shape_tensor_shape,
                            mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose2.input.extend([output_op_softmax])
                        op_transpose2.input.extend([shape_tensor_name2])
                        op_transpose2.output.extend(op.output)
                        op_transpose2.output_shape.add()
                        op_transpose2.output_shape[0].dims.extend(output_shape)

                        self._maceOpArray = np.append(self._maceOpArray,op_transpose2)
                        self.remove_op_with_name(op_name)

    def split_LogSoftmax(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                xi = op.input[0]
                op_name = op.name
                slice_point_enable = False
                inputTensor = self.find_tensor_by_name(xi)
                inputShape = self.get_shape_by_name(xi)
                output_shape = op.output_shape[0].dims[:]
                arg = op.arg
                shape_tensor_data = [x for x in range(0, len(inputShape))]

                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'axis':
                        arg[i].i = len(output_shape) + arg[i].i if arg[i].i < 0 else arg[i].i
                        axis= arg[i].i
                        mace_check(axis < len(output_shape) and axis >= 0,
                                " axis must be within length of input dims\n")

                # SGS Softmax only do softmax on innermost dim, so need to transpose according to axis param

                # last axis will be keeped at innermost dim. No need for special treatment
                if len(inputShape) == 4:
                    # 1\axis-dim is inner-dim,no need to transpose
                    if axis == len(inputShape) -1:
                        # ADD SOFTMAX
                        op_softmax= self._SGSModel.op.add()
                        op_softmax.name = op_name + '_softmax'
                        op_softmax.type = 'SOFTMAX'
                        op_softmax.input.extend([xi])
                        output_op_softmax = op_softmax.name + '_output'
                        op_softmax.output.extend([output_op_softmax])
                        op_softmax.output_shape.add()
                        op_softmax.output_shape[0].dims.extend(output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_softmax)
                        # ADD LOG
                        op_log = self._SGSModel.op.add()
                        op_log.name = op_name + '_log'
                        op_log.type = 'LOG'
                        op_log.input.extend([output_op_softmax])
                        op_log.output.extend(op_.output)
                        op_log.output_shape.add()
                        op_log.output_shape[0].dims.extend(output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_log)
                    # 2\need transpose axis-dim to inner-dim
                    elif axis == 1:
                        self.OpInputInsertTranspose(op_)
                        xi = op_.input[0]
                        if len(output_shape) == 4:
                            output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]

                        # ADD SOFTMAX
                        op_softmax= self._SGSModel.op.add()
                        op_softmax.name = op_name + '_softmax'
                        op_softmax.type = 'SOFTMAX'
                        op_softmax.input.extend([xi])
                        output_op_softmax = op_softmax.name + '_output'
                        op_softmax.output.extend([output_op_softmax])
                        op_softmax.output_shape.add()
                        op_softmax.output_shape[0].dims.extend(output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_softmax)
                        # ADD LOG
                        op_log = self._SGSModel.op.add()
                        op_log.name = op_name + '_log'
                        op_log.type = 'LOG'
                        op_log.input.extend([output_op_softmax])
                        op_log.output.extend(op_.output)
                        op_log.output_shape.add()
                        op_log.output_shape[0].dims.extend(output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_log)
                        self.OpOutputInsertTranspose(op_log)
                    # 3\For other cases, transpose dim pointed by arg axis
                    else:
                        temp = shape_tensor_data[axis]
                        shape_tensor_data[axis] = shape_tensor_data[-1]
                        shape_tensor_data[-1] = temp

                        # creat transpose for switch axis to innermost dim
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = 'SGS_' + op_name + '_transpose'
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shape1'
                        shape_tensor_shape = [len(inputShape)]
                        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                            mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([xi])
                        op_transpose.input.extend([shape_tensor_name])
                        output_op_transpose = op_transpose.name + '_output1'
                        op_transpose.output.extend([output_op_transpose])
                        tmp_shape = [x for x in range(0, len(inputShape))]
                        for i in six.moves.range(len(inputShape)):
                            if i == axis:
                                tmp_shape[i] = inputShape[-1]
                            elif i == len(inputShape) -1:
                                tmp_shape[i] = inputShape[axis]
                            else:
                                tmp_shape[i] = inputShape[i]

                        op_transpose.output_shape.add()
                        op_transpose.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                        # ADD SOFTMAX
                        op_softmax= self._SGSModel.op.add()
                        op_softmax.name = op_name + '_softmax'
                        op_softmax.type = 'SOFTMAX'
                        op_softmax.input.extend([output_op_transpose])
                        output_op_softmax = op_softmax.name + '_output'
                        op_softmax.output.extend([output_op_softmax])
                        op_softmax.output_shape.add()
                        op_softmax.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_softmax)

                        # ADD LOG
                        op_log = self._SGSModel.op.add()
                        op_log.name = op_name + '_log'
                        op_log.type = 'LOG'
                        op_log.input.extend([output_op_softmax])
                        output_op_log = op_log.name + '_output'
                        op_log.output.extend([output_op_log])
                        op_log.output_shape.add()
                        op_log.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_log)

                        # creat transpose for switch axis to innermost dim to resume tensor shape
                        op_transpose2 = self._SGSModel.op.add()
                        op_transpose2.name = op_name + '_transpose2'
                        op_transpose2.type = 'TRANSPOSE'
                        shape_tensor_name2 = op_transpose2.name + '_shape2'
                        shape_tensor_shape = [len(inputShape)]
                        self.add_tensor(self._SGSModel ,shape_tensor_name2, shape_tensor_shape,
                            mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose2.input.extend([output_op_softmax])
                        op_transpose2.input.extend([shape_tensor_name2])
                        op_transpose2.output.extend(op.output)
                        op_transpose2.output_shape.add()
                        op_transpose2.output_shape[0].dims.extend(output_shape)

                        self._maceOpArray = np.append(self._maceOpArray,op_transpose2)
                        self.remove_op_with_name(op_name)
                else:
                    # ADD SOFTMAX
                    op_softmax= self._SGSModel.op.add()
                    op_softmax.name = op_name + '_softmax'
                    op_softmax.type = 'SOFTMAX'
                    op_softmax.input.extend([xi])
                    output_op_softmax = op_softmax.name + '_output'
                    op_softmax.output.extend([output_op_softmax])
                    op_softmax.output_shape.add()
                    op_softmax.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_softmax)
                    # ADD LOG
                    op_log = self._SGSModel.op.add()
                    op_log.name = op_name + '_log'
                    op_log.type = 'LOG'
                    op_log.input.extend([output_op_softmax])
                    op_log.output.extend(op_.output)
                    op_log.output_shape.add()
                    op_log.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_log)

    def split_Softplus(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_name = op_.name
            op_.type = 'CUSTOM'
            op_.name = 'Softplus' + op_name
            self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Transpose(self, op):
        # unTransposed
        op_name = op.name
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        input_shape = self.get_shape_by_name(xi)
        intputShapeSize = len(input_shape)
        output_shape = op.output_shape[0].dims

        for op_ in self._SGSModel.op:
            if op_ == op:
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_dims_str:
                        dims = arg[i].ints
                shape_tensor_name = op_.name + '_shape'
                shape_tensor_data = dims
                shape_tensor_shape = [len(dims)]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                mace_pb2.DT_INT32, shape_tensor_data)
                op_.input.extend([shape_tensor_name])
                op_.type = "TRANSPOSE"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Atan(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
          if op_ == op:
            self.OpInputInsertTranspose(op_)
            op_.name = 'Atan' + op.name
            op_.type = 'CUSTOM'
            self._maceOpArray = np.append(self._maceOpArray,op_)
            self.OpOutputInsertTranspose(op_)

    def split_Tile(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
          if op_ == op:
            self.OpInputInsertTranspose(op_)
            op_.type = "TILE"
            self._maceOpArray = np.append(self._maceOpArray,op_)
            self.OpOutputInsertTranspose(op_)

    def split_Log(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
          if op_ == op:
            self.OpInputInsertTranspose(op_)
            op_.type = "LOG"
            self._maceOpArray = np.append(self._maceOpArray,op_)
            self.OpOutputInsertTranspose(op_)

    def split_Less(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
          if op_ == op:
            self.OpInputInsertTranspose(op_)
            op_.type = "LESS"
            self._maceOpArray = np.append(self._maceOpArray,op_)
            self.OpOutputInsertTranspose(op_)

    def split_Not(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            self.OpInputInsertTranspose(op_)
            op_.type = "LOGICAL_NOT"
            self._maceOpArray = np.append(self._maceOpArray,op_)
            self.OpOutputInsertTranspose(op_)

    def split_And(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            self.OpInputInsertTranspose(op_)
            op_.type = "LOGICAL_AND"
            self._maceOpArray = np.append(self._maceOpArray,op_)
            self.OpOutputInsertTranspose(op_)

    def split_Upsample3d(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                xi = op_.input[0]
                yi = op_.output[0]
                input_shape = self.get_shape_by_name(xi)
                output_shape = self.get_shape_by_name(yi)
                op_name = op_.name + '#SGS_UPSAMPLE'
                # create  transpose op
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '#transpose'
                op_transpose.type = 'TRANSPOSE'
                op_transpose.input.extend([xi])
                # set transpose's output_shape
                transpose_output_tensor_name = op_transpose.name + '_output_shape'
                transpose_output_tensor_data = [0, 2, 3, 4, 1]  # n,w,c
                transpose_output_tensor_shape = [5]
                self.add_tensor(self._SGSModel, transpose_output_tensor_name, transpose_output_tensor_shape,
                                mace_pb2.DT_INT32, transpose_output_tensor_data)
                op_transpose.input.extend([transpose_output_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                trans_in = [input_shape[0],input_shape[2],input_shape[3],input_shape[4],input_shape[1]]
                op_transpose.output_shape[0].dims.extend(trans_in)
                self._maceOpArray = np.append(self._maceOpArray, op_transpose)

                scale_data = [(a / b) for a, b in zip(output_shape, input_shape)]
                mace_check(scale_data[2] == scale_data[3] == scale_data[4],"Scale value must be same!")
                # check support limit:
                # (1) [D,H,W] scale value must be same;
                # (2) scale value must be an integer greater than 1.
                for i in six.moves.range(len(scale_data)):
                    if scale_data[i] < 1:
                        mace_check(False,"Scale value must be greater than 1!")
                    decimal_part = str(scale_data[i]).split('.')[-1]
                    for s in decimal_part:
                        if s != '0':
                            mace_check(False,"Scale value must be integer!")
                            break

                scale_data = [round(a / b) for a, b in zip(output_shape, input_shape)]
                out_data = [(a * b) for a, b in zip(scale_data, trans_in)]
                #creat tile
                op_tile = self._SGSModel.op.add()
                op_tile.name = op_name + '_tile'
                op_tile.type = 'TILE'
                multiples_tensor_name = op_tile.name + '_multiples'
                multiples_tensor_data = scale_data
                multiples_tensor_shape = [5]
                self.add_tensor(self._SGSModel ,multiples_tensor_name, multiples_tensor_shape,
                    mace_pb2.DT_INT32, multiples_tensor_data)
                op_tile.input.extend([output_op_transpose])
                op_tile.input.extend([multiples_tensor_name])
                output_op_tile = op_tile.name + '_output'
                op_tile.output.extend([output_op_tile])
                op_tile.output_shape.add()
                op_tile.output_shape[0].dims.extend(out_data)
                self._maceOpArray = np.append(self._maceOpArray,op_tile)

                #creat reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape#1'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_tile])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [output_shape[0],output_shape[2],output_shape[3],output_shape[4],output_shape[1]]
                reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend([output_shape[0],output_shape[2],output_shape[3],output_shape[4],output_shape[1]])
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                # create  transpose op
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '#transpose1'
                op_transpose.type = 'TRANSPOSE'
                op_transpose.input.extend([output_op_reshape])
                # set transpose's output_shape
                transpose_output_tensor_name = op_transpose.name + '_output_shape'
                transpose_output_tensor_data = [0,4,1,2,3]  # n,w,c
                transpose_output_tensor_shape = [5]
                self.add_tensor(self._SGSModel, transpose_output_tensor_name, transpose_output_tensor_shape,
                                mace_pb2.DT_INT32, transpose_output_tensor_data)
                op_transpose.input.extend([transpose_output_tensor_name])
                op_transpose.output[:] =  op_.output[:]
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(output_shape)
                self._maceOpArray = np.append(self._maceOpArray, op_transpose)

    def split_Upsample(self, op):
        # Transposed
        # only support 4 dims case
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                if len(self.get_shape_by_name(op.input[0])) == 5:
                    return self.split_Upsample3d(op)
                else:
                    mace_check(len(self.get_shape_by_name(op.input[0])) == 4,"SGS not support yet!")
                [n,c,h,w] = op_.output_shape[0].dims[:]
                [ni,hi,wi,ci] = self.get_shape_by_name(op_.input[0])
                op_name = op_.name
                xi = op_.input[0]#NHWC in interpreter
                scale = 1
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'scale':
                        scale = arg[i].i

                #creat tile
                op_tile = self._SGSModel.op.add()
                op_tile.name = op_name + '_tile'
                op_tile.type = 'TILE'
                multiples_tensor_name = op_tile.name + '_multiples'
                multiples_tensor_data = [1,1,scale,scale]
                multiples_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,multiples_tensor_name, multiples_tensor_shape,
                    mace_pb2.DT_INT32, multiples_tensor_data)
                op_tile.input.extend([xi])
                op_tile.input.extend([multiples_tensor_name])
                output_op_tile = op_tile.name + '_output'
                op_tile.output.extend([output_op_tile])
                op_tile.output_shape.add()
                op_tile.output_shape[0].dims.extend([ni,hi,wi*scale,ci*scale])
                self._maceOpArray = np.append(self._maceOpArray,op_tile)

                #creat reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape#1'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_tile])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [n,h,w,c]
                reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape.output[:] =  op_.output[:]
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend([n,h,w,c])
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                self.OpOutputInsertTranspose(op_reshape)
                #remove op
                self.remove_op_with_name(op_name)


    def split_Threshold(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "GREATER_EQUAL"
                self._maceOpArray = np.append(self._maceOpArray,op_)
                self.OpOutputInsertTranspose(op_)

    def split_ResizeBilinear(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "RESIZE_BILINEAR"
                #del op_.input[1]
                self._maceOpArray = np.append(self._maceOpArray,op_)
                self.OpOutputInsertTranspose(op_)

    def split_ResizeNearestNeighbor(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "RESIZE_NEAREST_NEIGHBOR"
                #del op_.input[1]
                self._maceOpArray = np.append(self._maceOpArray,op_)
                self.OpOutputInsertTranspose(op_)

    def split_SpaceToDepth(self, op):
        # unTransposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                xi = op_.input[0]
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'block_size':
                        block_size = arg[i].i
                        mace_check((block_size == 2),"only support block_size num is 2 yet")
                mace_check(len(self.get_shape_by_name(xi)) == 4,"SGS not support yet!")
                [n,c,h,w] = self.get_shape_by_name(xi)
                c1 = int(c*block_size*block_size)
                w1 = w//block_size
                h1 = h//block_size
                op_name = op_.name
                # add reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([xi])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [n,h1,block_size,w1,block_size,c]
                reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend([n,h1,block_size,w1,block_size,c])
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                # create transpose
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = 'SGS_' + op_name + '_transpose#1'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0, 1, 3, 2, 4, 5]
                shape_tensor_shape = [6]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                    mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([output_op_reshape])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend([n,h1,w1,block_size,block_size,c])
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                # create reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape#1'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_transpose])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [n, h1, w1, c1]
                reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape.output[:] =  op.output[:]
                op_reshape.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                self.remove_op_with_name(op_name)

    def split_GreaterOrEqual(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "GREATER_EQUAL"
                self._maceOpArray = np.append(self._maceOpArray, op_)
                self.OpOutputInsertTranspose(op_)

    def split_Cos(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "COS"
                self._maceOpArray = np.append(self._maceOpArray, op_)
                self.OpOutputInsertTranspose(op_)

    def split_Sin(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "SIN"
                self._maceOpArray = np.append(self._maceOpArray, op_)
                self.OpOutputInsertTranspose(op_)

    def split_SGS_SSD_Postprocess(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CUSTOM"
            self._maceOpArray = np.append(self._maceOpArray,op_)
        pass
    def split_SGS_YoloV2_Postprocess(self, op):
     for op_ in self._SGSModel.op:
       if op_ == op:
         op_.type = "CUSTOM"
         self._maceOpArray = np.append(self._maceOpArray,op_)
     pass

    def split_SGS_YoloV3_Postprocess(self, op):
     for op_ in self._SGSModel.op:
       if op_ == op:
         op_.type = "CUSTOM"
         self._maceOpArray = np.append(self._maceOpArray,op_)
     pass
    def split_SGS_LanceNet_Postprocess(self, op):
     for op_ in self._SGSModel.op:
       if op_ == op:
         op_.type = "CUSTOM"
         self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_SGS_FDA_Postprocess(self, op):
     for op_ in self._SGSModel.op:
       if op_ == op:
         op_.type = "CUSTOM"
         self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_SGS_CAFFE_SSD_Postprocess(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CUSTOM"
            self._maceOpArray = np.append(self._maceOpArray,op_)
        split_QIRQuantize
    def split_QIRQuantize(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CUSTOM"
            op_.name = 'QIRQuantize' + op.name
            self._maceOpArray = np.append(self._maceOpArray,op_)
