import numpy as np
import six
import math
import pdb
import copy
from mace.proto import mace_pb2
from mace.python.tools.convert_util import mace_check
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.sgs_caffe import SGSModel_genAnchors
from mace.python.tools.sgs_caffe import SGSModel_prior_box
from third_party import tflite
from collections import Counter

class TransformToSchema(object):
    """A class for transform mace model to schema model.
    """

    def __init__(self,    model,inputName,inputShapeMap,outputName,input_pack_model,output_pack_model):
        self._splitOP_transform = {
            'Activation':self.split_Activation,
            'ArgMax':self.split_ArgMax,
            'Axpy':self.split_Axpy,
            'BatchNorm':self.split_BatchNorm,
            'Concat':self.split_Concat,
            'Conv2D':self.split_Conv2D,
            'Crop':self.split_Crop,
            'CReLU':self.split_CReLU,
            'ContinuationIndicator':self.split_ContinuationIndicator,
            'Clip':self.split_Clip,
            'ChannelShuffle':self.split_ChannelShuffle,
            'Deconv2D':self.split_Deconv2D,
            'DepthwiseConv2d':self.split_DepthwiseConv2d,
            'DepthwiseDeconv2d':self.split_Deconv2D,
            'Dropout':self.split_Dropout,
            'Eltwise':self.split_Eltwise,
            'FullyConnected':self.split_FullyConnected,
            'LSTM':self.split_LSTM,
            'Pooling':self.split_Pooling,
            'PriorBox':self.split_PriorBox,
            'Reshape':self.split_Reshape,
            'Slice': self.split_Slice,
            'Split': self.split_Split,
            'Softmax':self.split_Softmax,
            'Scale':self.split_Scale,
            'Transpose':self.split_Transpose,
            'Tile':self.split_Tile,
            'ROIPooling':self.split_ROIPooling,
            'Normalize':self.split_Normalize,
            'Reorg':self.split_Reorg,
            'Reverse':self.split_Reverse,
            'Upsample':self.split_Upsample,
            'Threshold':self.split_Threshold,
            'Power':self.split_Power,
            'SGS_SSD_Postprocess':self.split_SGS_CAFFE_SSD_Postprocess,
            'SGS_YoloV2_Postprocess':self.split_SGS_YoloV2_Postprocess,
            'SGS_YoloV3_Postprocess':self.split_SGS_YoloV3_Postprocess,
            'SGS_LanceNet_Postprocess':self.split_SGS_LanceNet_Postprocess,
            'SGS_FDA_Postprocess':self.split_SGS_FDA_Postprocess,
            'PriorBox_RFC':self.split_PriorBox_RFC,
        }
        self._inputShapeMap = inputShapeMap
        self._inputName = inputName
        self._SGSModel = model
        self._outputName = outputName
        self._input_pack_model_arrays = input_pack_model
        self._output_pack_model_arrays = output_pack_model
        self._oriModel = copy.deepcopy(model)
        self._maceOpArray = np.array([])
        self.name_shape_map = {}
        self.finetuneArray = []
        self._lstm_num = 0
        self._ConstList = []
        self._MultiConsumerConstList = []
        self._TransposedConstList = []

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
          shape = self._inputShapeMap[name]
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
        if input_tensor.int32_data != []:
            input_const_data = input_tensor.int32_data
        else:
            input_const_data = input_tensor.float_data
        if len(input_const_data) != 0:
            data = np.array(input_const_data)
            data = data.reshape(input_shape)
            data = data.transpose(0,2,3,1)
            data = list(data.flat)
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
                                    input_tensor.data_type, data)
                else:
                    self._TransposedConstList.append(input_name)
                    ConstNameCounter = dict(Counter(self._TransposedConstList))
                    num = str(ConstNameCounter[input_name]-1)
                    new_input_name = input_name_str + num + '_#sgsnhwc_' + op_name
                    self.add_tensor(self._SGSModel ,new_input_name, new_input_shape,
                                    input_tensor.data_type, data)
            else:
                new_input_name = input_name_str
                self.remove_tensor_by_name(input_name)
                self.add_tensor(self._SGSModel ,new_input_name, new_input_shape,
                                input_tensor.data_type, data)
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
                self._inputShapeMap[output_op_transpose] = ori_input_shape[:]

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
    def find_op_by_output_name(self,name):
        for op_ in self._SGSModel.op:
         for j,top in enumerate(op_.output):
           if name == top:
            return op_
        six.print_("can not find op by name: %s" % (name))
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

    def set_output_datatype_by_input(self, op, input_datatype):
        for outputName in op.output:
          outputTensor = self.find_tensor_by_name(outputName)
          outputTensor.data_format = input_datatype

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

    def check_NCHW_And_Change(self, input_name, op_name):
        output_shape = self.get_shape_by_name(input_name)
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = 'MACE_' + op_name + '_transpose'
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

    def split_Activation(self, op):
        # 'RELU'/'LEAKYRELU'/'PRELU'/'SIGMOID' "LOGISTIC"/'RELU6'/'TANH'，Transposed.
        leakyrelu_coefficient,isLeakRelu,isPRelu = 0,0,0
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                arg = op_.arg
                xi = op_.input[0]
                inputTensor = self.find_tensor_by_name (xi)
                self.set_output_datatype_by_input (op_, inputTensor.data_format)
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
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)
                        if a_type.decode() == 'SIGMOID':
                            self.OpInputInsertTranspose(op_)
                            op_.type = "LOGISTIC"
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)
                        if a_type.decode() == 'RELU6':
                            self.OpInputInsertTranspose(op_)
                            op_.type = "RELU6"
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)
                        if a_type.decode() == 'TANH':
                            self.OpInputInsertTranspose(op_)
                            op_.type = "TANH"
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            self.OpOutputInsertTranspose(op_)

    def split_ArgMax(self, op):
        # if keepdim,Transposed,else,Untransposed
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

    def split_Axpy(self, op):
        # Transposed
        '''
        * @param Formulation:
        *            F = a * X + Y
        *    Shape info:
        *            a:  N x C          --> bottom[0]
        *            X:  N x C x H x W  --> bottom[1]
        *            Y:  N x C x H x W  --> bottom[2]
        *            F:  N x C x H x W  --> top[0]
        '''
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_name = op_.name
                a = op_.input[0]
                X = op_.input[1]
                Y = op_.input[2]
                a_shape = self.get_shape_by_name(a)
                output_shape = op_.output_shape[0].dims
                if len(output_shape) == 4:
                    trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                else:
                    trans_output_shape =  output_shape
                #creat Mul
                op_mul = self._SGSModel.op.add()
                op_mul.name = op_name + '_MUL'
                op_mul.type = 'MUL'
                op_mul.input.extend([a])
                op_mul.input.extend([X])
                output_name_mul = op.input[1] + '_xx' # Just borrow the old name to create new tensor name,Otherwise, there is a possibility of duplicate names.
                op_mul.output.extend([output_name_mul])
                op_mul.output_shape.add()
                op_mul.output_shape[0].dims.extend(trans_output_shape)
                #tensor data is variable,so didn't creat new tensor
                self._maceOpArray = np.append(self._maceOpArray,op_mul)

                #creat ADD
                op_add = self._SGSModel.op.add()
                op_add.name = op_name + '_ADD'
                op_add.type = 'ADD'
                op_add.input.extend([output_name_mul])
                op_add.input.extend([Y])
                op_add.output[:] = op_.output[:]
                op_add.output_shape.add()
                op_add.output_shape[0].dims.extend(trans_output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_add)
                #remove axpy op
                self.remove_op_with_name(op_name)
                self.OpOutputInsertTranspose(op_add)

    def split_BatchNorm(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                ori_xi = op_.input[0]
                self.OpInputInsertTranspose(op_)
                # yi = xi * scale_value + offset_value
                name = op_.name
                xi = op_.input[0]

                scale_value = op_.input[1]
                offset_value = op_.input[2]
                output_shape = op_.output_shape[0].dims
                if len(output_shape) == 4:
                    trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                else:
                    trans_output_shape =  output_shape
                #creat Mul
                op_mul = self._SGSModel.op.add()
                op_mul.name = name + '_MUL'
                op_mul.type = 'MUL'
                op_mul.input.extend([xi])
                op_mul.input.extend([scale_value])
                if op_.output[0].count('_xx') == 2:
                    # in caffe_converter, add_layer use '_xx' suffix to rename caffe tensor if duplicated
                    # for match caffe (Conv -> BatchNrom -> Relu) pattern
                    output_name_mul = ori_xi + '_xx'
                else:
                    output_name_mul = name + '_MUL'
                op_mul.output.extend([output_name_mul])
                op_mul.output_shape.add()
                op_mul.output_shape[0].dims.extend(trans_output_shape)
                #tensor data is variable,so didn't creat new tensor
                self._maceOpArray = np.append(self._maceOpArray,op_mul)

                #creat ADD
                op_add = self._SGSModel.op.add()
                op_add.name = name + '_ADD'
                op_add.type = 'ADD'
                op_add.input.extend([output_name_mul])
                op_add.input.extend([offset_value])
                op_add.output[:] = op_.output[:]
                op_add.output_shape.add()
                op_add.output_shape[0].dims.extend(trans_output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_add)
                #remove BatchNorm op
                self.remove_op_with_name(name)
                self.OpOutputInsertTranspose(op_add)

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
                    paddingType_arg = op_.arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    paddingType_arg.i = tflite.Padding.Padding().CAFFE
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)
                else:
                    op_.name = 'GroupConv' + op_name
                    op_.type = 'CUSTOM'
                    paddingType_arg = op_.arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    paddingType_arg.i = tflite.Padding.Padding().CAFFE
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)

    def split_CReLU(self, op):
        # Transposed
        # (mul * -1 + leakRelu)    leakRelu
        #           |              |
        #           | - - - -|- - -|
        #                    |
        #                  concat
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_name = op_.name
                xi = op_.input[0]
                inputShape = self.get_shape_by_name(xi)
                slope,axis = 0.0,1
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'negative_slope':
                        slope= arg[i].f
                    elif name == 'concat_axis':
                        arg[i].i = len(inputShape) + arg[i].i if arg[i].i < 0 else arg[i].i
                        axis = arg[i].i
                if len(op.output_shape[0].dims) == 4:#nchw -> nhwc
                    if axis == 1:
                        axis = 3
                    elif axis == 2:
                        axis = 1
                    elif axis == 3:
                        axis = 2
                # creat mul
                op_mul = self._SGSModel.op.add()
                op_mul.name = op_name + '_MUL'
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
                op_mul.output_shape[0].dims.extend(inputShape)
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
                op_lkyRelu.output_shape[0].dims.extend(inputShape)
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
                op_lkyRelu.output_shape[0].dims.extend(inputShape)
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
                op_concat.output[:] = op_.output[:]
                output_shape = op_.output_shape[0].dims
                if len(output_shape) == 4:
                    trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                else:
                    trans_output_shape = output_shape
                op_concat.output_shape.add()
                op_concat.output_shape[0].dims.extend(trans_output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_concat)
                self.OpOutputInsertTranspose(op_concat)
                self.remove_op_with_name(op_name)

    def split_ContinuationIndicator(self, op):
        op_name = op.name
        [t,b] = op.output_shape[0].dims[:]
        output_name = op.output[0]
        output_tensor = self.find_tensor_by_name(output_name)
        value = np.ones(t*b,dtype=np.float32).reshape(t,b)
        value[0,0] = 0.0
        output_tensor.data_type = mace_pb2.DT_FLOAT
        output_tensor.float_data.extend(value.flatten().tolist())

    def split_Crop(self, op):
        # Untransposed
        op_name = op.name
        input_data_name = op.input[0]
        output_shape = op.output_shape[0].dims[:]
        arg = op.arg
        offset = []

        # convert Crop args(axis, offset) to Slice args(begin)
        # Slice args(size) refers to output_shape
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == 'axis':
                axis= arg[i].i
                axis = len(output_shape) + axis if axis < 0 else axis
            elif name == 'offset':
                offset = arg[i].ints

        begin_slice = [0 for i in six.moves.range(len(output_shape))]
        offset_list = [0 for i in six.moves.range(len(output_shape))]
        if len(offset) == 1:
            offset_list = [offset[0] for i in six.moves.range(len(output_shape))]
        elif offset != []:
            for i in six.moves.range(len(offset)):
                offset_list[len(offset_list)-1-i] = offset[len(offset)-1-i]
        for i in six.moves.range(len(begin_slice)):
            if i >= axis:
                begin_slice[i] = offset_list[i]

        # creat slice
        op_slice = self._SGSModel.op.add()
        op_slice.name = op_name + '_slice'
        op_slice.type = 'SLICE'
        begin_tensor_name = op_slice.name + '_begin'
        begin_tensor_data = begin_slice[:]
        begin_tensor_shape = [len(output_shape)]
        self.add_tensor(self._SGSModel ,begin_tensor_name, begin_tensor_shape,
            mace_pb2.DT_INT32, begin_tensor_data)
        size_tensor_name = op_slice.name + '_size'
        size_tensor_data = op.output_shape[0].dims[:]
        size_tensor_shape = [len(output_shape)]
        self.add_tensor(self._SGSModel ,size_tensor_name, size_tensor_shape,
            mace_pb2.DT_INT32, size_tensor_data)
        op_slice.input.extend([input_data_name])
        op_slice.input.extend([begin_tensor_name])
        op_slice.input.extend([size_tensor_name])
        op_slice.output[:] = op.output[:]
        op_slice.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_slice)
        self.remove_op_with_name(op_name)

    def split_Clip(self, op):
        # Untransposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "RELU6"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Deconv2D(self, op):
        # Transposed
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
                        paddingL,paddingR,paddingT,paddingB= arg[i].ints
                    elif name == MaceKeyword.mace_strides_str:
                        strideH,strideW = arg[i].ints
                    elif name == MaceKeyword.mace_dilations_str:
                        dilationH,dilationW = arg[i].ints
                    elif name == 'group':
                        group = arg[i].i
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
                if group == c == nk and ck == 1 and group != 1:
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

                def computeOutSidePadding(input_shape,output_shape,strideH,strideW,kernel_shape):
                    [ni,ci,hi,wi] = input_shape
                    [n,c,h,w] = output_shape
                    input_pading_shape_H = hi + (hi - 1)*(strideH - 1)
                    input_pading_shape_W = wi + (wi - 1)*(strideW - 1)
                    round_func = math.floor
                    pad_h = round_func((h - 1 - input_pading_shape_H + kernel_shape[0])/2)
                    pad_w = round_func((w - 1 - input_pading_shape_W + kernel_shape[1])/2)
                    dilation_output_h = input_pading_shape_H + 2 * pad_h
                    dilation_output_w = input_pading_shape_W + 2 * pad_w
                    return pad_h,pad_w,dilation_output_h,dilation_output_w

                pad_h,pad_w,dilation_output_h,dilation_output_w = computeOutSidePadding(input_shape,output_shape,strideH,strideW,filter_tensor.dims[2:])
                #                #
                # creat dilation #
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
                op_dilation.input.extend([output_op_transpose])
                op_dilation.input.extend([outside_pad_name])
                op_dilation.input.extend([inside_pad_name])
                output_name_dilation = op_.name + '_dilation' + '_output'
                op_dilation.output.extend([output_name_dilation])
                op_dilation.output_shape.add()
                op_dilation.output_shape[0].dims.extend([ni,dilation_output_h,dilation_output_w,ci])
                self._maceOpArray = np.append(self._maceOpArray,op_dilation)

                if depthwise:
                    #                       #
                    # creat depthwiseconv   #
                    #                       #
                    op_dep_conv = self._SGSModel.op.add()
                    op_dep_conv.name = op_name + '_depthwise_conv'
                    op_dep_conv.type = 'DEPTHWISE_CONV_2D'
                    strides_arg = op_dep_conv.arg.add()
                    strides_arg.name = 'strides'
                    strides_arg.ints.extend([1,1])
                    padding_arg = op_dep_conv.arg.add()
                    padding_arg.name = 'padding_values'
                    padding_arg.ints.extend([0,0,0,0])
                    #add padding_type
                    paddingType_arg = op_dep_conv.arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    paddingType_arg.i = tflite.Padding.Padding().CAFFE
                    op_dep_conv.input.extend([output_name_dilation])
                    depth_filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(1,do,hk,wk)
                    depth_filter_data = np.transpose(depth_filter_data,(0,2,3,1))
                    filter_tensor.float_data[:] = depth_filter_data.flat
                    filter_tensor.dims[:] = [1,hk,wk,do]
                    op_dep_conv.input.extend([filter_name])
                    op_dep_conv.input.extend([biase_name])
                    op_dep_conv.output.extend(op.output)
                    op_dep_conv.output_shape.add()
                    output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                    op_dep_conv.output_shape[0].dims.extend(output_shape)
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
                    paddingType_arg.i = tflite.Padding.Padding().CAFFE
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
                    # creat conv2d   #
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
                    paddingType_arg.i = tflite.Padding.Padding().CAFFE
                    op_conv.input.extend([output_name_dilation])
                    depth_filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(nk,ck,hk,wk)
                    depth_filter_data = np.transpose(depth_filter_data,(0,2,3,1))
                    filter_tensor.float_data[:] = depth_filter_data.flat
                    filter_tensor.dims[:] = [nk,hk,wk,ck]
                    op_conv.input.extend([filter_name])
                    op_conv.input.extend([biase_name])
                    op_conv.output.extend(op.output)
                    #output_op_dep_conv = op_dep_conv.name + '_output'
                    op_conv.output_shape.add()
                    output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                    op_conv.output_shape[0].dims.extend(output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_conv)
                    self.OpOutputInsertTranspose(op_conv)
                self.remove_op_with_name(op_name)

    def split_DepthwiseConv2d(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
           if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "DEPTHWISE_CONV_2D"
                #add padding_type
                paddingType_arg = op_.arg.add()
                paddingType_arg.name = MaceKeyword.mace_padding_str
                paddingType_arg.i = tflite.Padding.Padding().CAFFE
                self._maceOpArray = np.append(self._maceOpArray,op_)
                self.OpOutputInsertTranspose(op_)


    def split_Dropout(self, op):
        # Untransposed
        op_name = op.name
        xi = op.input[0]
        xi_shape = self.get_shape_by_name (xi)
        out_shape = op.output_shape[0].dims[:]
        op_output_shape = copy.deepcopy(out_shape)
        # add reshape
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '_reshape'
        op_reshape.type = 'RESHAPE'
        op_reshape.input.extend([xi])
        reshape_output_tensor_name = op_reshape.name + '_output_shape'
        reshape_output_tensor_data = op_output_shape
        reshape_output_tensor_shape = [len(op_output_shape)]
        self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
        op_reshape.input.extend([reshape_output_tensor_name])
        op_reshape.output[:] = op.output[:]
        op_reshape.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_reshape)
        self.remove_op_with_name(op_name)

    def split_Eltwise(self, op):
        # Transposed
        # if one of the input length is 4dim, will reshape it to 4dim and do transpose if another input lengt is not 4dim
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                arg = op_.arg
                isSub = False
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == "coeff":
                        if len(arg[i].floats)==2 and arg[i].floats[0] == 1 and arg[i].floats[1] == -1:
                            isSub = True
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_element_type_str:
                        pooling_type = arg[i].i
                        # 0:SUM 1:SUB 2:PROD 3:DIV 4:MIN 5:MAX 6:NEG 7:ABS 8:SQR_DIFF 9:POW 10:EQUAL 11:FLOOR_DIV
                        if pooling_type == 0:
                            if isSub:
                                op_.type = 'SUB'
                            else:
                                if len(op_.input) > 2:
                                    op_.type = 'ADD_N'
                                else:
                                    op_.type = 'ADD'
                        elif pooling_type == 2:
                            op_.type = 'MUL'
                        elif pooling_type == 5:
                            op_.type = 'MAXIMUM'
                        else:
                            mace_check(False, "Does not support this eltwise op type %s" % op_name)

                if len(op_.input) > 1 :
                    # convert const tensors' shape:
                    #       if one of input dim = 4,
                    #       another input dim < 4,
                    #       < 4 need change to 4 dim.

                    # 1\change input if lhs is const
                    ori_x1 = op_.input[0]
                    ori_x2 = op_.input[1]
                    if self.is_const_tensor(ori_x1) == True and self.is_const_tensor(ori_x2) == False:
                        op_.input[1] = ori_x1
                        op_.input[0] = ori_x2
                    x1 = op_.input[0]
                    x2 = op_.input[1]
                    inputTensor1 = self.find_tensor_by_name(x1)
                    input_shape1 = self.get_shape_by_name(x1)
                    inputTensor2 = self.find_tensor_by_name(x2)
                    input_shape2 = self.get_shape_by_name(x2)
                    # # 2\create shape for scalar because scalar shape is []
                    # if self.is_scalar_tensor(x1):
                    #     input_shape1 = [1]
                    # if self.is_scalar_tensor(x2):
                    #     input_shape2 = [1]
                    # 3\convert const tensors' shape which dim < 4
                    needReshape = False
                    if len(input_shape1) == 4 or len(input_shape2) == 4:
                        needReshape = True
                    for inputName in op_.input:
                        if needReshape:
                            if self.is_const_tensor(inputName) == True:
                                tensor = self.find_tensor_by_name(inputName)
                                input_shape = self.get_shape_by_name(inputName)
                                # find scalar tensor:scalar shape is [],need change to [1]
                                if len(tensor.dims) == 0:
                                    input_shape = [1]
                                elif len(tensor.dims) == 1 and tensor.dims[0] == 1:
                                    input_shape = [1]
                                # reshep != 4dim to 4dim
                                if len(input_shape) != 4:
                                    num = 4 - len(input_shape)
                                    for i in range(num):
                                        input_shape.insert(0,1)
                                    inputTensor = self.find_tensor_by_name(inputName)
                                    # if self.is_scalar_tensor(inputName):
                                    #     for j in six.moves.range(len(inputTensor.dims)):
                                    #         inputTensor.dims.pop()
                                    #     inputTensor.dims.extend(input_shape)
                            elif self.is_const_tensor(inputName) == False:
                                tensor = self.find_tensor_by_name(inputName)
                                input_shape = self.get_shape_by_name(inputName)
                                if len(input_shape) != 4:
                                    mace_check(False,str(op_name) + ":" + "only support both input shapes are 4-dimension if inputs are both variable tensor!")

                    if len(inputTensor1.dims) == 4 or len(inputTensor2.dims) == 4:
                        self.OpInputInsertTranspose(op_)
                        self._maceOpArray = np.append(self._maceOpArray, op_)
                        self.OpOutputInsertTranspose(op_)
                    else:
                        self._maceOpArray = np.append(self._maceOpArray, op_)
                else:
                    self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_FullyConnected(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                arg = op_.arg
                op_name = op_.name
                xi = op_.input[0]
                weight = op_.input[1]
                bias = op_.input[2]
                input_shape = self.get_shape_by_name(xi)
                weight_shape = self.get_shape_by_name(weight)
                # find input op and input op input_shape
                if len(input_shape) == 4 and (input_shape[3] != weight_shape[3]):
                    #case for lenet
                    [n,h,w,c] = input_shape
                    # add reshape
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape'
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
                    op_reshape.output_shape[0].dims.extend([n,1,1,1*c*h*w])#n,c,h,w
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                    #creat ori fullconnect conv
                    op_conv = self._SGSModel.op.add()
                    op_conv.name = op_name + '_fully_connet'
                    op_conv.type = 'FULLY_CONNECTED'
                    tensor_weight = self.find_tensor_by_name(weight)
                    temp_data = np.array(tensor_weight.float_data).reshape(op_.output_shape[0].dims[1],1,c,h,w)
                    temp_data = temp_data.transpose(0,1,3,4,2)
                    del tensor_weight.float_data[:]
                    tensor_weight.float_data.extend(list(temp_data.flat))
                    op_conv.input.extend([output_op_reshape])
                    op_conv.input.extend([weight])
                    op_conv.input.extend([bias])
                    op_conv.output[:] = op_.output[:]
                    output_shape = op_.output_shape[0].dims
                    if len(output_shape) == 4:
                        trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                    else:
                        trans_output_shape = output_shape
                    op_conv.output_shape.add()
                    op_conv.output_shape[0].dims.extend(trans_output_shape)
                    #op_conv.output_shape[0].dims.extend([c,1,h,w])
                    self._maceOpArray = np.append(self._maceOpArray,op_conv)
                    self.OpOutputInsertTranspose(op_conv)
                    self.remove_op_with_name(op_name)
                else:
                    arg = op_.arg
                    axis = 1
                    output_shape = op_.output_shape[:]
                    op_input = op_.input[0]
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            axis_tensor_name = op_.name + '_axis'
                            axis = op_.arg[i].i
                            axis = len(input_shape) + axis if axis < 0 else axis
                    innershape = 1
                    for i in six.moves.range(len(input_shape)-axis):
                        innershape *= input_shape[axis+i]
                    mace_check(innershape == weight_shape[3],
                                "Input size incompatible with inner product parameters.")
                    if len(input_shape) != 4 and input_shape[-1] != innershape:
                        # add reshape
                        op_reshape = self._SGSModel.op.add()
                        op_reshape.name = op_name + '_reshape'
                        op_reshape.type = 'RESHAPE'
                        op_reshape.input.extend([xi])
                        reshape_data = []
                        for i in six.moves.range(len(input_shape)-axis):
                            reshape_data.append(input_shape[i])
                        reshape_data[axis] = innershape
                        reshape_output_tensor_name = op_reshape.name + '_output_shape'
                        reshape_output_tensor_data = reshape_data
                        reshape_output_tensor_shape = [axis+1]
                        self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                        mace_pb2.DT_INT32, reshape_output_tensor_data)
                        op_reshape.input.extend([reshape_output_tensor_name])
                        output_op_reshape = op_reshape.name + '_output'
                        op_reshape.output.extend([output_op_reshape])
                        op_reshape.output_shape.add()
                        op_reshape.output_shape[0].dims.extend(reshape_data)#n,c,h,w
                        self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                        #creat ori fullconnect
                        op_fully = self._SGSModel.op.add()
                        op_fully.name = op_name + '_fully_connet'
                        op_fully.type = 'FULLY_CONNECTED'
                        op_fully.input.extend([output_op_reshape])
                        op_fully.input.extend([weight])
                        op_fully.input.extend([bias])
                        output_op_fully = op_fully.name + '_output'
                        op_fully.output[:] = op_.output[:]
                        #op_fully.output_shape.add()
                        op_fully.output_shape.extend(output_shape)
                        #op_conv.output_shape[0].dims.extend([c,1,h,w])
                        self._maceOpArray = np.append(self._maceOpArray,op_fully)
                        self.remove_op_with_name(op_name)
                        return

                    op_.type = "FULLY_CONNECTED"
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    self.OpOutputInsertTranspose(op_)

    def split_LSTM(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_name = op_.name
                xi = op_.input[0]
                xi_shape = self.get_shape_by_name(xi)
                T = op_.input[1]
                T_shape = self.get_shape_by_name(T)
                num_output = op_.output_shape[0].dims[-1]
                T_tensor = self.find_tensor_by_name(T)
                T_time = T_tensor.dims[0]
                lstm_input_arrays = []
                lstm_data_input_tensor = xi
                shape_count = 1
                for i in six.moves.range(len(xi_shape)):
                    shape_count *= xi_shape[i]
                tmp = shape_count//xi_shape[0]
                if len(xi_shape) != 4 or xi_shape[1] !=1 or xi_shape[2] !=1: #chang 3 dim to 4 dim to fit sgs rule
                    # add reshape
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape'
                    op_reshape.type = 'RESHAPE'
                    op_reshape.input.extend([xi])
                    #tmp = shape_count//xi_shape[0]
                    reshape_output_tensor_name = op_reshape.name + '_output_shape'
                    reshape_output_tensor_data = [xi_shape[0],1,1,tmp]#n,h,w,c
                    reshape_output_tensor_shape = [4]
                    self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_output_tensor_data)
                    op_reshape.input.extend([reshape_output_tensor_name])
                    output_op_reshape = op_reshape.name + '_output'
                    op_reshape.output.extend([output_op_reshape])
                    op_reshape.output_shape.add()
                    op_reshape.output_shape[0].dims.extend([xi_shape[0],1,1,tmp])#n,h,w,c
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                    lstm_data_input_tensor = output_op_reshape

                name_prefix = "sgs_subnet_lstm" + str(self._lstm_num)
                # add h0
                h0_name = name_prefix + '_h0'
                h0_data = np.zeros(num_output)
                h0_shape = [1,1,1,num_output]
                self.add_tensor(self._SGSModel, h0_name, h0_shape,
                                mace_pb2.DT_FLOAT, h0_data)
                # add c0
                c0_name = name_prefix + '_c0'
                c0_data = np.zeros(num_output)
                c0_shape = [1,1,1,num_output]
                self.add_tensor(self._SGSModel, c0_name, c0_shape,
                                mace_pb2.DT_FLOAT, c0_data)

                # add offlie_size
                offline_size_name = name_prefix + '_size'
                offline_size_data = np.ones(2)
                offline_size_shape = [2]
                self.add_tensor(self._SGSModel, offline_size_name, offline_size_shape,
                                mace_pb2.DT_FLOAT, offline_size_data)

                #creat name-shape map for save to file
                input_output_map = {}
                input_output_map[name_prefix + '_input'] = [1,1,1,tmp]
                input_output_map[h0_name] = [1,1,1,num_output]
                input_output_map[c0_name] = [1,1,1,num_output]
                input_output_map[name_prefix + '_time'] = [1,1]
                input_output_map[name_prefix + '_output'] = op_.output_shape[0].dims[:]
                file_name = './lstm_data/input_output_shape#' + str(self._lstm_num) + '.npy'
                self._lstm_num += 1
                np.save(file_name,input_output_map, allow_pickle=True)
                #                #
                # creat SGS_LSTM #
                #                #
                op_SGS_LSTM = self._SGSModel.op.add()
                op_SGS_LSTM.name = 'SGS_LSTM'  + op_name
                op_SGS_LSTM.type = 'CUSTOM'
                #add inputs
                op_SGS_LSTM.input.extend([lstm_data_input_tensor])
                op_SGS_LSTM.input.extend([T])
                op_SGS_LSTM.input.extend([h0_name])
                op_SGS_LSTM.input.extend([c0_name])
                op_SGS_LSTM.input.extend([offline_size_name])
                #add outputs
                SGS_LSTM_output_array = []
                single_shape = [1,1,1,1]
                single_shape[3] = op.output_shape[0].dims[-1]
                for i in six.moves.range(T_time):
                    tmp_out_name = op_.name + '_output' + str(i)
                    op_SGS_LSTM.output.extend([tmp_out_name])
                    op_SGS_LSTM.output_shape.add()
                    op_SGS_LSTM.output_shape[i].dims.extend(single_shape)
                    SGS_LSTM_output_array.append(tmp_out_name)
                self._maceOpArray = np.append(self._maceOpArray,op_SGS_LSTM)

                # creat concat
                op_concat = self._SGSModel.op.add()
                op_concat.name = op_name + '_concat'
                op_concat.type = 'CONCATENATION'
                axis_arg = op_concat.arg.add()
                axis_arg.name = 'axis'
                axis_arg.i = 0
                for name in SGS_LSTM_output_array:
                    op_concat.input.extend([name])
                output_op_concat = op_concat.name + '_output'
                concat_shape = [1,1,1,1]
                concat_shape[0] = op_.output_shape[0].dims[0]
                concat_shape[3] = op_.output_shape[0].dims[-1]
                op_concat.output.extend([output_op_concat])
                op_concat.output_shape.add()
                op_concat.output_shape[0].dims.extend(concat_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_concat)

                # add reshape #chang 4 dim to 3 dim flow ori caffe
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_concat.name + '_reshape'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_concat])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                #reshape_output_tensor_data = [op.output_shape[0].dims[0],1,op.output_shape[0].dims[1]]#n,h,w,c
                reshape_output_tensor_data = op.output_shape[0].dims[:]
                reshape_output_tensor_shape = [3]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                #output_op_reshape = op_reshape.name + '_output'
                op_reshape.output[:] = op_.output[:]
                #op_reshape.output_shape.add()
                #op_reshape.output_shape[0].dims.extend([op.output_shape[0].dims[0],1,op.output_shape[0].dims[1]])#n,c,h,w
                op_reshape.output_shape.extend(op_.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                self.OpOutputInsertTranspose(op_reshape)
                self.remove_op_with_name(op_name)

    def split_Pooling(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                xi = op_.input[0]
                inputTensor = self.find_tensor_by_name(xi)
                kernel_h,kernel_w = -1,-1
                paddingL,paddingR,paddingT,paddingB = 0,0,0,0
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_pooling_type_str:
                        pooling_type = arg[i].i
                        if pooling_type == 1:
                            op_.type = 'AVERAGE_POOL_2D'
                        elif pooling_type == 2:
                            op_.type = 'MAX_POOL_2D'
                    elif name == MaceKeyword.mace_padding_values_str:
                        paddingL,paddingR,paddingT,paddingB = arg[i].ints
                    elif name == MaceKeyword.mace_global_pooling_str:
                        input_op_name = op_.input[0]
                        #find output_shape of bottom op
                        for input_op in self._SGSModel.op:
                            output_name_list = input_op.output
                            for output_name in output_name_list:
                                if output_name == input_op_name:
                                    kernel_h,kernel_w = input_op.output_shape[0].dims[1:3]
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_kernel_str:
                        if kernel_h != -1:
                            arg[i].ints[:] = []
                            arg[i].ints.extend([kernel_h,kernel_w])
                if op_.type == 'AVERAGE_POOL_2D' and  paddingL != 0:
                    [ni,hi,wi,ci] = self.get_shape_by_name(xi)
                    #add pad
                    op_pad = self._SGSModel.op.add()
                    op_pad.name = op.name + '_pad'
                    op_pad.type = 'PAD'
                    shape_tensor_name = op_pad.name + '_shape'
                    shape_tensor_data = [0,0,paddingT,paddingB,paddingL,paddingR,0,0]#[n,h,w,c]
                    shape_tensor_shape = [4,2]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                    mace_pb2.DT_INT32, shape_tensor_data)
                    op_pad.input.extend([xi])
                    op_pad.input.extend([shape_tensor_name])
                    output_op_pad = op_pad.name + '_output'
                    op_pad.output.extend([output_op_pad])
                    op_pad.output_shape.add()
                    output_shape_pad = [ni,int(hi+2*paddingT),int(wi+2*paddingL),ci]
                    op_pad.output_shape[0].dims.extend(output_shape_pad)
                    self._maceOpArray = np.append(self._maceOpArray,op_pad)
                    #modify average pooling
                    arg = op_.arg
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_padding_values_str:
                            arg[i].ints[:] = [0,0,0,0]
                    op_.input[0] = output_op_pad
                #add padding_type
                paddingType_arg = op_.arg.add()
                paddingType_arg.name = MaceKeyword.mace_padding_str
                paddingType_arg.i = tflite.Padding.Padding().CAFFE
                self._maceOpArray = np.append(self._maceOpArray,op_)
                self.OpOutputInsertTranspose(op_)

    def split_PriorBox(self, op):
        # unTransposed
        #init param
        max_sizes = []
        step_h,step_w,img_h,img_w =0,0,0,0
        op_name = op.name
        input_conv_name = op.input[0]
        data = op.input[1]
        [img_height,img_width] = self.get_shape_by_name(data)[2:]
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

            for name in op.input:
                for key,value in self._inputShapeMap.items():
                    if key == name:
                        [img_height,img_width]= value[2:]
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
        pass

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
                op_.input.extend([output_tensor_name])
                op_.type = "RESHAPE"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Slice(self, op):
        # unTransposed
        # caffe slice == tensorflow slice + tensorflow slice + ...
        xi = op.input[0]
        axis = 0
        slice_point_enable = False
        inputTensor = self.find_tensor_by_name(xi)
        inputShape = self.get_shape_by_name(xi)
        op_name = op.name
        arg = op.arg
        output_shape = op.output_shape[:]
        self.set_output_datatype_by_input (op, inputTensor.data_format)
        begin_data = list(np.zeros((len(inputShape)), dtype=np.int))
        use_slice = False

        for i in six.moves.range(len(op.output)):
            if self.is_tensor_detached(op.output[i]):
                use_slice = True

        # if one of op output tensors detached, cannot use split. Use slice instead.
        if use_slice:
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_axis_str:
                    arg[i].i = len(inputShape) + arg[i].i if arg[i].i < 0 else arg[i].i
                    axis = arg[i].i
                if name == "slice_point":
                    slice_point_enable = True
                    slice_point = arg[i].ints
                    slice_dims = copy.deepcopy(inputShape[axis])
                    slice_point_data = []
                    for i in six.moves.range(len(slice_point)+1):
                        if i == 0:
                            slice_point_data.append(slice_point[i])
                        elif i == len(slice_point):
                            slice_point_data.append(int(slice_dims - slice_point[i - 1]))
                        else:
                            slice_point_data.append(int(slice_point[i] - slice_point[i - 1]))
            #calu begin and size data
            begin_data = list(np.zeros((len(op.output_shape)), dtype=np.int))
            size_data = list(np.zeros((len(op.output_shape)), dtype=np.int))
            if slice_point_enable:
                size_data = slice_point_data
                for i in six.moves.range(len(op.output_shape)):
                    if i != 0:
                        begin_data[i] = slice_point[i-1]
            else:
                slice_num = inputShape[axis]//len(op.output_shape)
                for i in six.moves.range(len(op.output_shape)):
                    size_data[i] = slice_num
                    if i != 0:
                        begin_data[i] = slice_num
            #creat slice op
            for i in six.moves.range(len(op.output_shape)):
                op_slice = self._SGSModel.op.add()
                op_slice.name = op_name + '_slice#' + str(i)
                op_slice.type = 'SLICE'
                begin_tensor_name = op_slice.name + '_begin'
                begin_tensor_data = list(np.zeros((len(inputShape)), dtype=np.int))
                axis_begin = axis
                begin_tensor_data[axis_begin] = begin_data[i]
                begin_tensor_shape = [len(inputShape)]
                self.add_tensor(self._SGSModel ,begin_tensor_name, begin_tensor_shape,
                                mace_pb2.DT_INT32, begin_tensor_data)
                size_tensor_name = op_slice.name + '_size'
                size_tensor_data = copy.deepcopy(inputShape)
                size_tensor_data[axis] = size_data[i]
                size_tensor_shape = [len(inputShape)]
                self.add_tensor(self._SGSModel ,size_tensor_name, size_tensor_shape,
                                mace_pb2.DT_INT32, size_tensor_data)
                op_slice.input.extend([xi])
                op_slice.input.extend([begin_tensor_name])
                op_slice.input.extend([size_tensor_name])
                op_slice.output.extend([op.output[i]])
                op_slice.output_shape.add()
                op_slice.output_shape[0].dims.extend(size_tensor_data)
                #op_slice.output[:] =  op.output[:]
                #op_slice.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_slice)
            #remove slice op
            self.remove_op_with_name(name)
        else:
            for op_ in self._SGSModel.op:
                if op_ == op:
                    arg = op_.arg
                    output_shape = op_.output_shape[:]
                    op_input = op_.input[0]
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            arg[i].i = len(inputShape) + arg[i].i if arg[i].i < 0 else arg[i].i
                            axis = arg[i].i
                            axis_tensor_name = op_.name + '_axis'
                            axis_tensor_data = [op_.arg[i].i]
                            axis_tensor_shape = [1]
                            self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                                            mace_pb2.DT_INT32, axis_tensor_data)
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == "slice_point":
                            slice_point_enable = True
                            slice_point = arg[i].ints
                            slice_dims = copy.deepcopy(inputShape[axis])
                            slice_point_data = []
                            for i in six.moves.range(len(slice_point)+1):
                                if i == 0:
                                    slice_point_data.append(slice_point[i])
                                elif i == len(slice_point):
                                    slice_point_data.append(int(slice_dims - slice_point[i - 1]))
                                else:
                                    slice_point_data.append(int(slice_point[i] - slice_point[i - 1]))
                            slice_point_name = op_.name + '_slice_point'
                            slice_point_shape = [len(slice_point)+1]
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
                    numSplits_arg = op_.arg.add()
                    numSplits_arg.name = MaceKeyword.mace_num_split_str
                    numSplits_arg.i = len(op_.output_shape)
                    self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Split(self, op):
        # unTransposed
        op_name = op.name
        isOutPutNode = False
        for topName in op.output:
            for outputName in self._outputName:
                if outputName == topName:
                        isOutPutNode = True
        if isOutPutNode:
            xi = op.input[0]
            # creat concat
            op_concat = self._SGSModel.op.add()
            op_concat.name = op_name + '_concat'
            op_concat.type = 'CONCATENATION'
            axis_arg = op_concat.arg.add()
            axis_arg.name = 'axis'
            axis_arg.i = 0
            for i in six.moves.range(len(op.output)):
                op_concat.input.extend([xi])
            output_op_concat = op_concat.name + '_output'
            op_concat.output.extend([output_op_concat])
            op_concat.output_shape.add()
            bottom_shape = self.get_shape_by_name(xi)
            bottom_shape[0] = bottom_shape[0] * len(op.output)
            op_concat.output_shape[0].dims.extend(bottom_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_concat)

            # creat split
            op_split = self._SGSModel.op.add()
            op_split.name = op_name + '_split'
            op_split.type = 'SPLIT'
            axis_tensor_name = op_split.name + '_axis'
            axis_data = [0]
            axis_shape = [1]
            self.add_tensor(self._SGSModel, axis_tensor_name, axis_shape,
                            mace_pb2.DT_INT32, axis_data)
            op_split.input.extend([axis_tensor_name])
            op_split.input.extend([output_op_concat])
            for topName in op.output:
                op_split.output.extend([topName])
                op_split.output_shape.add()
                op_split.output_shape[0].dims.extend(self.get_shape_by_name(xi))
            numSplits_arg = op_split.arg.add()
            numSplits_arg.name = MaceKeyword.mace_num_split_str
            numSplits_arg.i = len(op_split.output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_split)
            self.remove_op_with_name(op_name)
        else:
            bottom_top_map = {}
            bottom_top_map["bottom"] = op.input[0]
            for i,name in enumerate(op.output):
                bottom_top_map["top"+str(i)] = name
            self.finetuneArray.append(bottom_top_map)
            self.remove_op_with_name(op_name)

    def split_Softmax(self, op):
        # Transposed no matter dim length.but if axis is inner dim will not do transpose no matter dim length.
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
                        op_transpose.name = 'MACE_' + op_name + '_transpose'
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
                        op_transpose.name = 'MACE_' + op_name + '_transpose'
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

    def split_Scale(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                # check input shape
                if len(op_.input) > 1 :
                    # convert const tensors' shape:
                    #       if one of input dim = 4,(variable)
                    #       another input dim < 4,(const)
                    #       < 4 need change to 4 dim. This step has been completed in the previous stage
                    #
                    # check variable tensors' shape:
                    #       if one of input dim = 4,(variable)
                    #       another input dim < 4,(variable)
                    #       do reshape for this case
                    #
                    # TIPS:
                    # (1) Caffe can't do broadcast,  So the dimensions on the same axis are either the same or have no value
                    #     support[1,3,2,4] * [1,3,2] / unspport[1,3,2,4] * [1,3,2,1]
                    # (2) Scale layer supports cases where input0 is longer than or equal to input1, unsupport input0 dims < input1
                    #     support[1,3,2,4] * [1,3,2] / unspport[1,3,2] * [1,3,2,4]
                    x1 = op_.input[0]
                    x2 = op_.input[1]
                    inputTensor1 = self.find_tensor_by_name(x1)
                    input_shape1 = self.get_shape_by_name(x1)
                    inputTensor2 = self.find_tensor_by_name(x2)
                    input_shape2 = self.get_shape_by_name(x2)
                    needReshape = False
                    if len(input_shape1) == 4 and len(input_shape2) != 4:
                        needReshape = True
                    for inputName in op_.input:
                        if needReshape:
                            if self.is_const_tensor(inputName) == False:
                                tensor = self.find_tensor_by_name(inputName)
                                input_shape = self.get_shape_by_name(inputName)
                                if len(input_shape) != 4:
                                    #mace_check(False,str(op_.type) + ":" + "only support both input shapes are 4-dimension if inputs are both variable tensor!")
                                    num = 4
                                    tmp_shape = [1,1,1,1]
                                    tmp_shape[:len(input_shape)] = input_shape[:]
                                    shape = tmp_shape
                                    op_reshape = self._SGSModel.op.add()
                                    op_reshape.name = op_.name + '_reshape'
                                    op_reshape.type = 'RESHAPE'
                                    reshape_output_tensor_name = op_reshape.name + '_output_shape'
                                    reshape_output_tensor_data = shape
                                    reshape_output_tensor_shape = [len(shape)]
                                    self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                                    op_reshape.input.extend([inputName])
                                    op_reshape.input.extend([reshape_output_tensor_name])
                                    op_reshape_output_name = op_reshape.name + '_output'
                                    op_reshape.output.extend([op_reshape_output_name])
                                    op_reshape.output_shape.add()
                                    op_reshape.output_shape[0].dims.extend(shape)
                                    self.add_tensor(self._SGSModel, op_reshape_output_name, shape,
                                                mace_pb2.DT_FLOAT, None)
                                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                                    op_.input[1] = op_reshape_output_name
                                    self.name_shape_map[op_reshape_output_name] = shape

                self.OpInputInsertTranspose(op_)
                op_name = op_.name
                xi = op_.input[0]
                scale = op_.input[1]
                scale_shape = self.get_shape_by_name(scale)
                arg = op_.arg
                output_shape = op_.output_shape[0].dims
                if len(output_shape) == 4:
                    trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                else:
                    trans_output_shape = output_shape
                axis = 1

                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_axis_str:
                        axis = arg[i].i
                if len(op_.input) == 3: #has bias
                    bias = op_.input[2]
                    op_mul = self._SGSModel.op.add()
                    op_mul.name = op_name + '_mul'
                    op_mul.type = 'MUL'
                    op_mul.input.extend([xi,scale])
                    op_mul_output = op_mul.name + '_output'
                    op_mul.output.extend([op_mul_output])
                    op_mul.output_shape.add()
                    op_mul.output_shape[0].dims.extend(trans_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_mul)
                    op_add = self._SGSModel.op.add()
                    op_add.name = name + '_add'
                    op_add.type = 'ADD'
                    op_add.input.extend([op_mul_output])
                    op_add.input.extend([bias])
                    op_add.output[:] = op_.output[:]
                    op_add.output_shape.add()
                    op_add.output_shape[0].dims.extend(trans_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_add)
                    self.OpOutputInsertTranspose(op_add)
                else:
                    op_mul = self._SGSModel.op.add()
                    op_mul.name = op_name + '_mul'
                    op_mul.type = 'MUL'
                    op_mul.input.extend([xi])
                    op_mul.input.extend([scale])
                    op_mul.output[:] =  op_.output[:]
                    op_mul.output_shape.add()
                    op_mul.output_shape[0].dims.extend(trans_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_mul)
                    self.OpOutputInsertTranspose(op_mul)
                self.remove_op_with_name(op_name)

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
                # shape_tensor_name = op_.name + '_shape'
                # shape_tensor_data = dims
                # shape_tensor_shape = [len(dims)]
                # self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                #                 mace_pb2.DT_INT32, shape_tensor_data)
                # op_.input.extend([shape_tensor_name])
                op_.type = "TRANSPOSE"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Tile(self, op):
        # unTransposed
        op_name = op.name
        xi = op.input[0]
        arg = op.arg
        output_shape = op.output_shape[0].dims
        op_input = op.input[0]
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == MaceKeyword.mace_axis_str:
                axis = arg[i].i
            if name == 'tiles':
                tiles = arg[i].i
        #creat TILE
        op_tile = self._SGSModel.op.add()
        op_tile.name = op_name
        op_tile.type = 'TILE'
        multiples_tensor_name = op_tile.name + '_multiples'
        tmp = np.ones(len(output_shape),dtype=np.int8)
        tmp[axis] =  tiles
        multiples_data = tmp
        multiples_shape = [len(output_shape)]
        self.add_tensor(self._SGSModel ,multiples_tensor_name, multiples_shape,
                      mace_pb2.DT_INT32, multiples_data)
        op_tile.input.extend([xi])
        op_tile.input.extend([multiples_tensor_name])

        op_tile.output[:] =  op.output[:]
        op_tile.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_tile)
        self.remove_op_with_name(op_name)

    def split_ROIPooling(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_.type = "CUSTOM"
                op_.name = "SGS_ROIPooling" + op.name
                self._maceOpArray = np.append(self._maceOpArray,op_)
                self.OpOutputInsertTranspose(op_)

    def split_Normalize(self, op):
        # unTransposed
        across_spatial,channel_shared = 1,1
        mace_check(len(op.output_shape[0].dims[:]) == 4,"SGS not suppotr yet!")
        [n,c,h,w] = op.output_shape[0].dims[:]
        op_name = op.name
        xi = op.input[0]
        scale = op.input[1]
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'across_spatial':
            across_spatial = arg[i].i
          elif name == 'channel_shared':
            channel_shared = arg[i].i
          elif name == 'eps':
            eps = arg[i].f

        #creat mul : xi * xi
        op_mul = self._SGSModel.op.add()
        op_mul.name = op_name + '_MUL'
        op_mul.type = 'MUL'
        op_mul.input.extend([xi,xi])
        op_mul_output = op_mul.name + '_output'
        op_mul.output.extend([op_mul_output])
        op_mul.output_shape.extend(op.output_shape)
        #tensor data is variable,so didn't creat new tensor
        self._maceOpArray = np.append(self._maceOpArray,op_mul)

        if across_spatial == 1:

          #creat sum
          #or replace by reshpe(1*(h*w*c)*1*1) + conv
          op_sum = self._SGSModel.op.add()
          op_sum.name = op_name + '_sum'
          op_sum.type = 'SUM'
          op_sum.input.extend([op_mul_output])
          output_op_sum = op_sum.name + '_output'
          op_sum.output.extend([output_op_sum])
          op_sum.output_shape.add()
          op_sum.output_shape[0].dims.extend([1])
          self._maceOpArray = np.append(self._maceOpArray,op_sum)

          #creat sqrt
          op_sqrt = self._SGSModel.op.add()
          op_sqrt.name = op_name + '_sqrt'
          op_sqrt.type = 'SQRT'
          op_sqrt.input.extend([output_op_sum])
          output_op_sqrt = op_sqrt.name + '_output'
          op_sqrt.output.extend([output_op_sqrt])
          op_sqrt.output_shape.add()
          op_sqrt.output_shape[0].dims.extend([1])
          self._maceOpArray = np.append(self._maceOpArray,op_sqrt)

          #creat add
          #add eps invoid divisor is 0
          op_add = self._SGSModel.op.add()
          op_add.name = op_name + '_add'
          op_add.type = 'ADD'
          eps_tensor_name = op_add.name + '_eps'
          eps_data = [eps]
          eps_shape = [1]
          self.add_tensor(self._SGSModel ,eps_tensor_name, eps_shape,
              mace_pb2.DT_FLOAT, eps_data)
          op_add.input.extend([output_op_sqrt])
          op_add.input.extend([eps_tensor_name])
          output_op_add = op_add.name + '_output'
          op_add.output.extend([output_op_add])
          op_add.output_shape.add()
          op_add.output_shape[0].dims.extend([1])
          self._maceOpArray = np.append(self._maceOpArray,op_add)


          #creat div
          op_div = self._SGSModel.op.add()
          op_div.name = op_name + '_div'
          op_div.type = 'DIV'
          op_div.input.extend([xi])
          op_div.input.extend([output_op_add])
          output_op_div = op_div.name + '_output'
          op_div.output.extend([output_op_div])
          #op_div.output[:] =  op.output[:]
          op_div.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_div)
        else:
          #creat conv
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
          paddingType_arg.i = tflite.Padding.Padding().CAFFE
          filter_tensor_name = op_conv.name + '_filter'
          filter_data = np.ones((1,c,1,1),dtype = np.float32)
          filter_shape = [1,c,1,1]
          self.add_tensor(self._SGSModel, filter_tensor_name, filter_shape,
                        mace_pb2.DT_FLOAT, filter_data.flatten().tolist())
          bias_data = np.zeros(1)
            # caffe of old version has 4-dimension bias, so reshape it
            # to single dimension
          bias_tensor_name = op_conv.name + '_bias'
          self.add_tensor(self._SGSModel,bias_tensor_name, bias_data.shape,
                          mace_pb2.DT_FLOAT,bias_data)
          op_conv.input.extend([op_mul_output])
          op_conv.input.extend([filter_tensor_name])
          op_conv.input.extend([bias_tensor_name])
          output_op_conv = op_conv.name + '_output'
          op_conv.output.extend([output_op_conv])
          op_conv.output_shape.add()
          op_conv.output_shape[0].dims.extend([1,1,h,w])
          self._maceOpArray = np.append(self._maceOpArray,op_conv)

          #creat sqrt
          op_sqrt = self._SGSModel.op.add()
          op_sqrt.name = op_name + '_sqrt'
          op_sqrt.type = 'SQRT'
          op_sqrt.input.extend([output_op_conv])
          output_op_sqrt = op_sqrt.name + '_output'
          op_sqrt.output_shape.add()
          op_sqrt.output_shape[0].dims.extend([h,w])
          op_sqrt.output.extend([output_op_sqrt])

          self._maceOpArray = np.append(self._maceOpArray,op_sqrt)

          #creat TILE
          op_tile = self._SGSModel.op.add()
          op_tile.name = op_name + '_tile'
          op_tile.type = 'TILE'
          multiples_tensor_name = op_conv.name + '_multiples'
          multiples_data = [c,1]
          multiples_shape = [1,2]
          self.add_tensor(self._SGSModel ,multiples_tensor_name, multiples_shape,
                        mace_pb2.DT_INT32, multiples_data)
          op_tile.input.extend([output_op_sqrt])
          op_tile.input.extend([multiples_tensor_name])
          output_op_tile = op_tile.name + '_output'
          op_tile.output_shape.add()
          op_tile.output_shape[0].dims.extend([c*h,w])
          op_tile.output.extend([output_op_tile])
          self._maceOpArray = np.append(self._maceOpArray,op_tile)

          #creat reshape
          op_reshape = self._SGSModel.op.add()
          op_reshape.name = op_name + '_reshape'
          op_reshape.type = 'RESHAPE'
          reshape_output_tensor_name = op_reshape.name + '_output_shape'
          reshape_output_tensor_data = [1,h,c,w]
          reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
          self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                          mace_pb2.DT_INT32, reshape_output_tensor_data)
          op_reshape.input.extend([output_op_tile])
          op_reshape.input.extend([reshape_output_tensor_name])
          output_op_reshape = op_reshape.name + '_output'
          op_reshape.output_shape.add()
          op_reshape.output.extend([output_op_reshape])
          op_reshape.output_shape[0].dims.extend([1,h,c,w])
          self._maceOpArray = np.append(self._maceOpArray,op_reshape)

          #creat transpose
          op_transpose = self._SGSModel.op.add()
          op_transpose.name = op_name + '_transpose'
          op_transpose.type = 'TRANSPOSE'
          shape_tensor_name = op_transpose.name + '_shape'
          shape_tensor_data = [0,2,3,1]
          shape_tensor_shape = [4]
          self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
              mace_pb2.DT_INT32, shape_tensor_data)
          op_transpose.input.extend([output_op_reshape])
          op_transpose.input.extend([shape_tensor_name])
          output_op_transpose = op_transpose.name + '_output'
          op_transpose.output.extend([output_op_transpose])
          op_transpose.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_transpose)

          #creat add
          #add eps invoid divisor is 0
          op_add = self._SGSModel.op.add()
          op_add.name = op_name + '_add'
          op_add.type = 'ADD'
          eps_tensor_name = op_add.name + '_eps'
          eps_data = [eps]
          eps_shape = [1]
          self.add_tensor(self._SGSModel ,eps_tensor_name, eps_shape,
              mace_pb2.DT_FLOAT, eps_data)
          op_add.input.extend([output_op_transpose])
          op_add.input.extend([eps_tensor_name])
          output_op_add = op_add.name + '_output'
          op_add.output.extend([output_op_add])
          op_add.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_add)


          #creat div
          op_div = self._SGSModel.op.add()
          op_div.name = op_name + '_div'
          op_div.type = 'DIV'
          op_div.input.extend([xi])
          op_div.input.extend([output_op_add])
          output_op_div = op_div.name + '_output'
          op_div.output.extend([output_op_div])
          #op_div.output[:] =  op.output[:]
          op_div.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_div)
        if channel_shared == 1:
          #creat mul : top_data * scale[0]
          op_mul = self._SGSModel.op.add()
          op_mul.name = op_name + '_mul#1'
          op_mul.type = 'MUL'

          op_mul.input.extend([output_op_div])
          op_mul.input.extend([scale])

          op_mul.output[:] =  op.output[:]
          op_mul.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_mul)
        else:

          #creat reshape
          op_reshape = self._SGSModel.op.add()
          op_reshape.name = op_name + '_reshape#1'
          op_reshape.type = 'RESHAPE'
          reshape_output_tensor_name = op_reshape.name + '_output_shape'
          reshape_output_tensor_data = [c,1,1,1]
          reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
          self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                          mace_pb2.DT_INT32, reshape_output_tensor_data)
          op_reshape.input.extend([scale])
          op_reshape.input.extend([reshape_output_tensor_name])
          output_op_reshape = op_reshape.name + '_output'
          op_reshape.output.extend([output_op_reshape])
          op_reshape.output_shape.add()
          op_reshape.output_shape[0].dims.extend([c,1,1,1])
          self._maceOpArray = np.append(self._maceOpArray,op_reshape)

          #creat conv
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
          paddingType_arg.i = tflite.Padding.Padding().CAFFE
          spatial_multiplier_name = op_conv.name + '_spatial_multiplier'
          spatial_multiplier_data = np.ones((1,1,h,w),dtype = np.float32)
          spatial_multiplier_shape = spatial_multiplier_data.shape
          self.add_tensor(self._SGSModel ,spatial_multiplier_name, spatial_multiplier_shape,
                        mace_pb2.DT_FLOAT, spatial_multiplier_data.flatten().tolist())
          bias_data = np.zeros(c)
          bias_tensor_name = op_conv.name + '_bias'
          self.add_tensor(self._SGSModel, bias_tensor_name, bias_data.shape,
                          mace_pb2.DT_FLOAT,bias_data)

          op_conv.input.extend([spatial_multiplier_name])
          op_conv.input.extend([output_op_reshape])
          op_conv.input.extend([bias_tensor_name])
          output_op_conv = op_conv.name + '_output'
          op_conv.output.extend([output_op_conv])
          op_conv.output_shape.extend(op.output_shape)
          #op_conv.output_shape[0].dims.extend([c,1,h,w])
          self._maceOpArray = np.append(self._maceOpArray,op_conv)


          #creat mul
          op_mul = self._SGSModel.op.add()
          op_mul.name = op_name + '_mul#1'
          op_mul.type = 'MUL'
          op_mul.input.extend([output_op_div])
          op_mul.input.extend([output_op_conv])
          op_mul.output[:] =  op.output[:]
          op_mul.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_mul)

        #remove Normalize op
        self.remove_op_with_name(op_name)

    def split_Reorg(self, op):
        # unTransposed
        #only support strid is 2
        op_name = op.name
        xi = op.input[0]
        mace_check(len(self.get_shape_by_name(xi)) == 4,"SGS not support yet!")
        [n,c,h,w] = op.output_shape[0].dims[:]
        c = int(c/4)
        #creat reshape
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '_reshape'
        op_reshape.type = 'RESHAPE'
        op_reshape.input.extend([xi])
        reshape_output_tensor_name = op_reshape.name + '_output_shape'
        reshape_output_tensor_data = [n,int(c/2),h,2,2,w,2]
        reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
        self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
        op_reshape.input.extend([reshape_output_tensor_name])
        output_op_reshape = op_reshape.name + '_output'
        op_reshape.output.extend([output_op_reshape])
        op_reshape.output_shape.add()
        op_reshape.output_shape[0].dims.extend([n,int(c/2),h,2,2,w,2])
        self._maceOpArray = np.append(self._maceOpArray,op_reshape)

        #creat transpose
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = op_name + '_transpose#1'
        op_transpose.type = 'TRANSPOSE'
        shape_tensor_name = op_transpose.name + '_shape'
        shape_tensor_data = [0,3,6,1,2,4,5]
        shape_tensor_shape = [7]
        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
            mace_pb2.DT_INT32, shape_tensor_data)
        op_transpose.input.extend([output_op_reshape])
        op_transpose.input.extend([shape_tensor_name])
        output_op_transpose = op_transpose.name + '_output'
        op_transpose.output.extend([output_op_transpose])
        op_transpose.output_shape.add()
        op_transpose.output_shape[0].dims.extend([n,2,2,int(c/2),h,2,w])
        self._maceOpArray = np.append(self._maceOpArray,op_transpose)

        #creat reshape
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '_reshape#1'
        op_reshape.type = 'RESHAPE'
        op_reshape.input.extend([output_op_transpose])
        reshape_output_tensor_name = op_reshape.name + '_output_shape'
        reshape_output_tensor_data = [n,w,int(4*c),h]
        reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
        self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
        op_reshape.input.extend([reshape_output_tensor_name])

        op_reshape.output[:] = op.output[:]
        op_reshape.output_shape.add()
        op_reshape.output_shape[0].dims.extend([n,w,int(4*c),h])
        self._maceOpArray = np.append(self._maceOpArray,op_reshape)

        #remove op
        self.remove_op_with_name(op_name)

    def split_Reverse(self, op):
        # unTransposed
        op_name = op.name
        xi = op.input[0]#NHWC in interpreter
        input_shape = self.get_shape_by_name(xi)
        axis = 0
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_axis_str:
            arg[i].i = len(input_shape) + arg[i].i if arg[i].i < 0 else arg[i].i
            axis = arg[i].i

        # creat STRIDED_SLICE
        op_stride_slice = self._SGSModel.op.add()
        op_stride_slice.name = op_name + '_stride_slice'
        op_stride_slice.type = 'STRIDED_SLICE'
        begin_tensor_name = op_stride_slice.name + '_begin'
        begin_tensor_data = [0 for i in six.moves.range(len(input_shape))]
        begin_tensor_data[axis] = -1
        begin_tensor_shape = [len(input_shape)]
        self.add_tensor(self._SGSModel ,begin_tensor_name, begin_tensor_shape,
            mace_pb2.DT_INT32, begin_tensor_data)
        end_tensor_name = op_stride_slice.name + '_end'
        end_tensor_data = copy.deepcopy(input_shape)
        end_tensor_data[axis] = (end_tensor_data[axis] + 1)* -1
        #end_tensor_data = [ni+1,hi+1,wi+1,-(ci+1)]
        end_tensor_shape = [len(end_tensor_data)]
        self.add_tensor(self._SGSModel ,end_tensor_name, end_tensor_shape,
            mace_pb2.DT_INT32, end_tensor_data)
        stride_tensor_name = op_stride_slice.name + '_stride'
        stride_tensor_data = [1 for i in six.moves.range(len(input_shape))]
        stride_tensor_data[axis] = -1
        #stride_tensor_data = [1,1,1,-1]
        stride_tensor_shape = [len(input_shape)]
        self.add_tensor(self._SGSModel ,stride_tensor_name, stride_tensor_shape,
            mace_pb2.DT_INT32, stride_tensor_data)

        op_stride_slice.input.extend([xi])
        op_stride_slice.input.extend([begin_tensor_name])
        op_stride_slice.input.extend([end_tensor_name])
        op_stride_slice.input.extend([stride_tensor_name])
        op_stride_slice.output[:] = op.output[:]
        op_stride_slice.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_stride_slice)
        self.remove_op_with_name(op_name)

    def split_Upsample(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                mace_check(len(self.get_shape_by_name(op_.input[0])) == 4,"SGS not support yet!")
                [n,c,h,w] = op_.output_shape[0].dims[:]
                [ni,hi,wi,ci] = self.get_shape_by_name(op_.input[0])
                op_name = op_.name
                output_shape = op_.output_shape[0].dims
                if len(output_shape) == 4:
                    trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                else:
                    trans_output_shape = output_shape
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
                op_reshape.output.extend([op_.output[0]])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(trans_output_shape)
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

    def split_Power(self, op):
        # Transposed
        # yi = xi * scale_value + offset_value
        for op_ in self._SGSModel.op:
            if op_ == op:
                self.OpInputInsertTranspose(op_)
                op_name = op_.name
                xi = op_.input[0]
                scale_value = op_.input[1]
                offset_value = op_.input[2]
                output_shape = op_.output_shape[0].dims
                if len(output_shape) == 4:
                    trans_output_shape = [output_shape[0],output_shape[2],output_shape[3],output_shape[1]]
                else:
                    trans_output_shape = output_shape
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == 'power':
                        power = arg[i].f
                        mace_check(power > 0,"power value not support negative yet.")
                integer_power = True
                decimal_part = str(power).split('.')[-1]
                for s in decimal_part:
                    if s != '0':
                        integer_power = False
                        break
                if  integer_power:
                    power = int(power)
                    #creat Mul
                    op_mul = self._SGSModel.op.add()
                    op_mul.name = op_name + '_MUL'
                    op_mul.type = 'MUL'
                    op_mul.input.extend([xi])
                    op_mul.input.extend([scale_value])
                    output_name_mul = op_mul.name + '_output'
                    op_mul.output.extend([output_name_mul])
                    op_mul.output_shape.add()
                    op_mul.output_shape[0].dims.extend(trans_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_mul)

                    #creat ADD
                    op_add = self._SGSModel.op.add()
                    op_add.name = op_name + '_ADD'
                    op_add.type = 'ADD'
                    op_add.input.extend([output_name_mul])
                    op_add.input.extend([offset_value])
                    if power == 1:
                        op_add.output.extend([op_.output[0]])
                        op_add.output_shape.add()
                        op_add.output_shape[0].dims.extend(trans_output_shape)
                    else:
                        output_name_add = op_add.name + '_output'
                        op_add.output.extend([output_name_add])
                        op_add.output_shape.add()
                        op_add.output_shape[0].dims.extend(trans_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_add)

                    loop_num = power - 1
                    loop = False
                    for i in six.moves.range(loop_num):
                        loop = True
                        #loop add mul
                        op_mul = self._SGSModel.op.add()
                        op_mul.name = op_name + '_MUL#' + str(i)
                        op_mul.type = 'MUL'
                        op_mul.input.extend([output_name_add])
                        if i == 0:
                            op_mul.input.extend([output_name_add])
                        else:
                            op_mul.input.extend([output_name_mul])
                        if i == (loop_num - 1):
                            op_mul.output[:] = op_.output[:]
                        else:
                            output_name_mul = op_mul.name + '_output#' + str(i)
                            op_mul.output.extend([output_name_mul])
                        op_mul.output_shape.add()
                        op_mul.output_shape[0].dims.extend(trans_output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_mul)
                    if loop == True:
                        self.OpOutputInsertTranspose(op_mul)
                    else:
                        self.OpOutputInsertTranspose(op_add)
                    self.remove_op_with_name(op_name)
                else:
                    mace_check(power == 0.5, "power value not support yet.")
                    if self.is_const_tensor(scale_value) == True:
                        if self.find_tensor_by_name(scale_value).int32_data != []:
                            scale_input_ori_data = float(self.find_tensor_by_name(scale_value).int32_data)
                        else:
                            scale_input_ori_data = self.find_tensor_by_name(scale_value).float_data
                    if self.is_const_tensor(offset_value) == True:
                        if self.find_tensor_by_name(offset_value).int32_data != []:
                            offset_input_ori_data = float(self.find_tensor_by_name(offset_value).int32_data)
                        else:
                            offset_input_ori_data = self.find_tensor_by_name(offset_value).float_data

                    if scale_input_ori_data == [1.0] and offset_input_ori_data == [0.0]:
                        op_.type = 'SQRT'
                        for i in six.moves.range(len(op_.input)):
                            if len(op_.input) > 1:
                                op_.input.pop()
                        self._maceOpArray = np.append(self._maceOpArray,op_)
                        self.OpOutputInsertTranspose(op_)
                    else:
                        #creat Mul
                        op_mul = self._SGSModel.op.add()
                        op_mul.name = op_name + '_MUL'
                        op_mul.type = 'MUL'
                        op_mul.input.extend([xi])
                        op_mul.input.extend([scale_value])
                        output_name_mul = op_mul.name + '_output'
                        op_mul.output.extend([output_name_mul])
                        op_mul.output_shape.add()
                        op_mul.output_shape[0].dims.extend(trans_output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_mul)

                        #creat ADD
                        op_add = self._SGSModel.op.add()
                        op_add.name = op_name + '_ADD'
                        op_add.type = 'ADD'
                        op_add.input.extend([output_name_mul])
                        op_add.input.extend([offset_value])
                        output_name_add = op_add.name + '_output'
                        op_add.output.extend([output_name_add])
                        op_add.output_shape.add()
                        op_add.output_shape[0].dims.extend(trans_output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_add)

                        #creat SQRT
                        op_sqrt = self._SGSModel.op.add()
                        op_sqrt.name = op_name + '_SQRT'
                        op_sqrt.type = 'SQRT'
                        op_sqrt.input.extend([output_name_add])
                        op_sqrt.output.extend([op_.output[0]])
                        op_sqrt.output_shape.add()
                        op_sqrt.output_shape[0].dims.extend(trans_output_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_sqrt)

                        self.OpOutputInsertTranspose(op_sqrt)
                        self.remove_op_with_name(op_name)

    def split_ChannelShuffle(self, op):
        # unTransposed
      # Calculated according to the NCHW data arrangement:
      # 1\transpose
      #   Rearranged back to NCHW
      # 2\reshape
      #   [n,c,h,w] -> [n,g,c/g,h,w]
      # 3\transpose
      #   [n,g,c/g,h,w] -> [n,c/g,g,h,w]
      # 4\reshape
      #   [n,c/g,g,h,w] -> [n,c,h,w]
      # 5\transpose to NHWC
      #   Rearranged as NHWC
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                xi = op_.input[0]
                [n,c,h,w] = inputShape = self.get_shape_by_name(xi) # channelShuufle Proposed in ShffleNet,used with conv,musut be 4dim
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == "group":
                        group = arg[i].i

                # 2\create reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape1'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([xi])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [n,group,c//group,h,w]
                reshape_output_tensor_shape = [5]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                output_op_reshape = op_reshape.name + '_output'
                op_reshape.output.extend([output_op_reshape])
                op_reshape.output_shape.add()
                op_shape = [n,group,c//group,h,w]
                op_reshape.output_shape[0].dims.extend([n,group,c//group,h,w])
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                #3\create transpose
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = op_name + '_transpose2'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0, 2, 1, 3, 4]
                shape_tensor_shape = [5]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([output_op_reshape])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                tmp_dim = [0, 2, 1, 3, 4]
                tmp_shape = [0,1,2,3,4]
                for i in six.moves.range(len(tmp_dim)):
                  tmp_shape[i] = op_shape[tmp_dim[i]]
                op_transpose.output_shape.add()
                op_transpose_shape = copy.deepcopy(tmp_shape)
                op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                # 4\create reshape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape2'
                op_reshape.type = 'RESHAPE'
                op_reshape.input.extend([output_op_transpose])
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = [n,c,h,w]
                reshape_output_tensor_shape = [4]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape.output[:] =  op_.output[:]
                op_reshape.output_shape.extend(op_.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                self.remove_op_with_name(op_name)


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
     pass

    def split_SGS_FDA_Postprocess(self, op):
     for op_ in self._SGSModel.op:
       if op_ == op:
         op_.type = "CUSTOM"
         self._maceOpArray = np.append(self._maceOpArray,op_)
     pass
    def split_SGS_CAFFE_SSD_Postprocess(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CUSTOM"
            self._maceOpArray = np.append(self._maceOpArray,op_)
        pass
