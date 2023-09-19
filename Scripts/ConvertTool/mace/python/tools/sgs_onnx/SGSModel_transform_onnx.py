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
            'And':self.split_And,
            'ArgMax':self.split_ArgMax,
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
            'Deconv2D':self.split_Deconv2D,
            'DepthwiseConv1d': self.split_DepthwiseConv1d,
            'DepthwiseConv2d':self.split_DepthwiseConv2d,
            'DepthwiseDeconv2d':self.split_Deconv2D,
            'Dropout':self.split_Dropout,
            'DepthToSpace':self.split_DepthToSpace,
            'Expand':self.split_Expand,
            'Eltwise':self.split_Eltwise,
            'Exp':self.split_Exp,
            'Erf':self.split_Erf,
            'FullyConnected':self.split_FullyConnected,
            'Gather':self.split_Gather,
            'HardSigmoid': self.split_HardSigmoid,
            'InstanceNorm': self.split_InstanceNorm,
            'LSTM':self.split_LSTM,
            'Less':self.split_Less,
            'GRU':self.split_GRU,
            'MatMul':self.split_MatMul,
            'Pad':self.split_Pad,
            'Pooling':self.split_Pooling,
            'PriorBox':self.split_PriorBox,
            'Reshape':self.split_Reshape,
            'Where':self.split_Where,
            'Greater': self.split_Greater,
            'GreaterOrEqual': self.split_GreaterOrEqual,
            'Elu': self.split_Elu,
            'ScatterND': self.split_scatternd,
            'Slice': self.split_Slice,
            'Split': self.split_Split,
            'Sin':self.split_Sin,
            'Softmax':self.split_Softmax,
            'Softplus':self.split_Softplus,
            'Transpose':self.split_Transpose,
            'Tile':self.split_Tile,
            'Normalize':self.split_Normalize,
            'Not':self.split_Not,
            'Reorg':self.split_Reorg,
            'ReduceL2':self.split_ReduceL2,
            'Upsample':self.split_Upsample,
            'Threshold':self.split_Threshold,
            'Reduce':self.split_Reduce,
            'Reduce_mean':self.split_Reduce_Mean,
            'ResizeBilinear':self.split_ResizeBilinear,
            'ResizeNearestNeighbor':self.split_ResizeNearestNeighbor,
            'SpaceToDepth':self.split_SpaceToDepth,
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

    def creatDynamicTensors(self, SGSModel):
        input_pack_model_name_arry = []
        for i in six.moves.range(len(self._input_pack_model_arrays)):
            if self._input_pack_model_arrays[i] == "NCHW":
                inputName = copy.deepcopy(self._inputName[i])
                input_pack_model_name_arry.append(inputName)
        #1.collect all tensor (include constant and virable tensor)
        outputname_list = []
        outputShape_list = []
        #add constant tensor
        for op in SGSModel.op:
          for outputTensor in op.output:
            outputname_list.append(outputTensor)
          for outputShape in op.output_shape:
            outputShape_list.append(outputShape.dims)
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
          if name in input_pack_model_name_arry:
             self.add_tensor(SGSModel, name, shape, mace_pb2.DT_FLOAT, None, mace_pb2.DT_NHWC)
          else:
            self.add_tensor(SGSModel, name, shape, mace_pb2.DT_FLOAT, None)
          self.name_shape_map[name] = shape

        for tensor in SGSModel.tensors:
          self.name_shape_map[tensor.name] = tensor.dims

    def convertNCHWOutputTensors(self):
        output_pack_model = []
        if self._output_pack_model_arrays != 'None':
            for i in six.moves.range(len(self._output_pack_model_arrays)):
                if self._output_pack_model_arrays[i] == "NCHW":
                    outputName = copy.deepcopy(self._outputName[i])
                    output_pack_model.append(outputName)
        # When model output tensor.data_format == mace_pb2.DT_NHWC,
        # tensor shape will keep as origin NCHW, no one will convert it to NHWC.
        # So insert Transpose before model output in order to convert NCHW tensor to NHWC manually
        model = copy.deepcopy(self._SGSModel)
        for op_ in model.op:
          oriOpOutput = copy.deepcopy(op_.output)
          for oriOpOutName in oriOpOutput:
            if oriOpOutName in self._outputName:
              opOutputTensor = self.find_tensor_by_name(oriOpOutName)
              oriOutputShape = self.get_shape_by_name(oriOpOutName)
              op_name = op_.name
              if len(opOutputTensor.dims)==4 and opOutputTensor.data_format == mace_pb2.DT_NHWC and oriOpOutName not in output_pack_model:
                # create new op output tensor replacing origin output tensor
                newOpOutName = op_name + '_' + oriOpOutName +'_newOutput'
                self.add_tensor(self._SGSModel ,newOpOutName, oriOutputShape,
                    opOutputTensor.data_type, None, mace_pb2.DT_NHWC)
                index = 0
                for outputName in op_.output:
                  if outputName != oriOpOutName:
                    index=index+1
                for op_real in self._SGSModel.op:
                  if op_real == op_:
                    op_real.output[index] = newOpOutName

                # add transpose before model output for nchw-nhwc
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = op_name + '_' + oriOpOutName +'_transpose'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0,2,3,1]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                    mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([newOpOutName])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
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
                opOutputTensor.data_format = mace_pb2.DT_NHWC

                for other_op in self._SGSModel.op:
                    oriOpInput = copy.deepcopy(other_op.input)
                    for i in six.moves.range(len(oriOpInput)):
                        if oriOpInput[i] == oriOpOutName:
                            other_op.input[i] = newOpOutName

    def infect_tensor_transpose_setting(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            inputTransSetting = mace_pb2.DT_NHWC
            for inputName in op_.input:
              inputTensor = self.find_tensor_by_name(inputName)
              if inputTensor.data_format == mace_pb2.DT_NCHW:
                inputTransSetting = mace_pb2.DT_NCHW
                break
            for outputName in op_.output:
              outputTensor = self.find_tensor_by_name(outputName)
              outputTensor.data_format = inputTransSetting
            break

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

    def set_output_datatype_by_input(self, op, input_datatype):
        for outputName in op.output:
          outputTensor = self.find_tensor_by_name(outputName)
          outputTensor.data_format = input_datatype

    def run(self):
        self.creatDynamicTensors(self._SGSModel)
        if self._input_pack_model_arrays != None:
            self.changeInputPackModel()
        for op in self._oriModel.op:
            type = op.type
            self.infect_tensor_transpose_setting(op)
            self._splitOP_transform[type](op)
        self.convertNCHWOutputTensors()
        #self.finetuneNet()
        if self._output_pack_model_arrays != None:
            self.changeOutputPackModel()
        return self._SGSModel,self._maceOpArray


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
        op_transpose.name = op_name + '_transpose'
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

    def create_Eltwise_op(self, opName, opType, outputTensorName, outputTensorShape):
        x1 = opName.input[0]
        x2 = opName.input[1]
        op_type = opType
        type_str = op_type.lower()
        op_name = opName.name
        op_elt = self._SGSModel.op.add()
        op_elt.name = op_name + type_str
        op_elt.type = op_type
        op_elt.input.extend([x1])
        op_elt.input.extend([x2])
        op_elt.output.extend([outputTensorName])
        op_elt.output_shape.add()
        op_elt.output_shape[0].dims.extend(outputTensorShape)
        self.add_tensor(self._SGSModel, outputTensorName, outputTensorShape, mace_pb2.DT_FLOAT,
                        None, data_fromat = mace_pb2.DT_NHWC)
        self._maceOpArray = np.append(self._maceOpArray,op_elt)

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

    def changeInputPackModel(self):
        #change input tensor to NCHW
        if self._input_pack_model_arrays != 'None':
            input_pack_model = []
            for i in six.moves.range(len(self._input_pack_model_arrays)):
                if self._input_pack_model_arrays[i] == "NCHW":
                    inputName = copy.deepcopy(self._inputName[i])
                    input_pack_model.append(inputName)

            model = copy.deepcopy(self._SGSModel)
            for i in six.moves.range(len(self._inputName)):
                input_tensor = self.find_tensor_by_name(self._inputName[i])
                ori_input_shape = self.get_shape_by_name(self._inputName[i])

                if len(ori_input_shape) == 4 and self._inputName[i] in input_pack_model:
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = self._inputName[i] + '_input_shadow'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = [0,2,3,1]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                    mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([self._inputName[i]])
                    op_transpose.input.extend([shape_tensor_name])
                    output_op_transpose = op_transpose.name + '_output'
                    op_transpose.output.extend([output_op_transpose])
                    op_transpose.output_shape.add()
                    op_transpose.output_shape[0].dims.extend(ori_input_shape)
                    self.add_tensor(self._SGSModel, output_op_transpose, ori_input_shape,
                                    mace_pb2.DT_FLOAT, None, mace_pb2.DT_NCHW)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    self.name_shape_map[output_op_transpose] = ori_input_shape[:]

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

    def changeOutputPackModel(self):
        #change output tensor to NCHW
        if self._output_pack_model_arrays != 'None':
            output_pack_model = []
            for i in six.moves.range(len(self._output_pack_model_arrays)):
                if self._output_pack_model_arrays[i] == "NCHW":
                    outputName = copy.deepcopy(self._outputName[i])
                    output_pack_model.append(outputName)
            fillterTensor = []
            for op in self._SGSModel.op:
                for topName in op.output:
                    for i,outputName in enumerate(output_pack_model):
                        if outputName == topName and (outputName not in fillterTensor):# find output op
                            fillterTensor.append(topName)
                            op_name = op.name
                            outputTensor = self.find_tensor_by_name(outputName)
                            output_shape = outputTensor.dims[:]
                            if len(output_shape) == 4 and outputTensor.data_format != mace_pb2.DT_NHWC : #change PackModel to caffe
                                shadowTensor_name = outputTensor.name + "_shadow"
                                shadowTensor_data = []
                                shadowTensor_shape = output_shape
                                self.add_tensor(self._SGSModel ,shadowTensor_name, shadowTensor_shape,
                                    outputTensor.data_type, shadowTensor_data)

                                for j in six.moves.range(len(op.output)):
                                  name = op.output[j]
                                  if name == outputTensor.name:
                                    op.output[j]  = shadowTensor_name

                                #for muti_output case
                                muti_output = False
                                for op_boy in self._SGSModel.op:
                                    for inputName in op_boy.input:
                                        if inputName == topName:
                                            muti_output = True
                                            op_boy.input.remove(inputName)
                                            op_boy.input.extend([shadowTensor_name])
                                # creat transpose for NHWC to NCHW in tflite model
                                #output_shape = self.get_shape_by_name(xi)
                                op_transpose = self._SGSModel.op.add()
                                op_transpose.name = op_name + '_transpose' + str(i)
                                op_transpose.type = 'TRANSPOSE'
                                shape_tensor_name = op_transpose.name + '_shape'
                                shape_tensor_data = [0,3,1,2]
                                shape_tensor_shape = [4]
                                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                    mace_pb2.DT_INT32, shape_tensor_data)
                                op_transpose.input.extend([shadowTensor_name])
                                op_transpose.input.extend([shape_tensor_name])
                                op_transpose.output.extend([outputName])
                                tmp_dim = [0,3,1,2]# NCHW --[0312]--> NWCH  -----> NCHW
                                tmp_shape = [0,1,2,3]
                                for i in six.moves.range(len(tmp_dim)):
                                    tmp_shape[i] = output_shape[tmp_dim[i]]
                                op_transpose.output_shape.add()
                                op_transpose.output_shape[0].dims.extend(tmp_shape)
                                #change output Tensor shape
                                del outputTensor.dims[:]
                                outputTensor.dims.extend(list(tmp_shape))
                                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                            elif len(output_shape) < 4: #reshape to 4 dim
                                shadowTensor_name = outputTensor.name + "_shadow" + str(i)
                                shadowTensor_data = []
                                shadowTensor_shape = output_shape
                                self.add_tensor(self._SGSModel ,shadowTensor_name, shadowTensor_shape,
                                    outputTensor.data_type, shadowTensor_data)

                                for j in six.moves.range(len(op.output)):
                                  name = op.output[j]
                                  if name == outputTensor.name:
                                    op.output[j]  = shadowTensor_name

                                #creat reshape
                                tmp_shape = [1,1,1,1]
                                output_len = len(output_shape)
                                for i in six.moves.range(output_len):
                                    tmp_shape[3-i] = output_shape[output_len-1-i]
                                op_reshape = self._SGSModel.op.add()
                                op_reshape.name = op_name + '_output_shadow_reshape' + str(i)
                                op_reshape.type = 'RESHAPE'
                                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                                reshape_output_tensor_data = tmp_shape
                                reshape_output_tensor_shape = [4]
                                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                                op_reshape.input.extend([shadowTensor_name])
                                op_reshape.input.extend([reshape_output_tensor_name])
                                op_reshape.output.extend([outputName])
                                op_reshape.output_shape.add()
                                op_reshape.output_shape[0].dims.extend(tmp_shape)
                                outputTensor.dims[:] = tmp_shape
                                outputTensor.data_format = mace_pb2.DT_NHWC
                                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

    def split_Activation(self, op):
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
                   op_.type = 'RELU'
                   self._maceOpArray = np.append(self._maceOpArray,op_)
                 if a_type.decode() == 'LEAKYRELU':
                   op_.type = "LEAKY_RELU"
                   self._maceOpArray = np.append(self._maceOpArray,op_)
                 if a_type.decode() == 'PRELU':
                   slope_name = op_.input[1]
                   slope = self.find_tensor_by_name(slope_name)
                   if len(slope.dims) == 3:
                     slope_shape = slope.dims
                     slope_data = np.array(slope.float_data).reshape(slope_shape)
                     slope_shape_c = slope_shape[0]
                     del slope_shape[0]
                     slope_shape.append(slope_shape_c)
                     slope_data = np.transpose(slope_data, (1, 2, 0))
                     slope.Clear()
                     slope.name = slope_name
                     slope.dims.extend(slope_shape)
                     slope.data_format = mace_pb2.DT_NHWC
                     slope.data_type = mace_pb2.DT_FLOAT
                     slope.float_data.extend(list(slope_data.flat))
                   op_.type = "PRELU"
                   self._maceOpArray = np.append(self._maceOpArray,op_)
                 if a_type.decode() == 'SIGMOID':
                   op_.type = "LOGISTIC"
                   self._maceOpArray = np.append(self._maceOpArray,op_)
                 if a_type.decode() == 'RELUX':
                   op_.type = "RELU6"
                   self._maceOpArray = np.append(self._maceOpArray,op_)
                 if a_type.decode() =='TANH':
                   op_.type = "TANH"
                   self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_ArgMax(self, op):
        for op_ in self._SGSModel.op:
           if op_ == op:
            axis = 1
            output_shape = op_.output_shape[:]
            inputShape = self.get_shape_by_name(op_.input[0])
            op_.type = "ARG_MAX"
            arg = op_.arg
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == MaceKeyword.mace_axis_str:
                arg[i].i = self.get_axes_by_shape (arg[i].i, len(inputShape))
                axis = arg[i].i
            #add axis tensor into input arrays
            axis_tensor_name = op.name + '_axis'
            axis_data = [axis]
            axis_data_shape = [1]
            self.add_tensor(self._SGSModel, axis_tensor_name, axis_data_shape,
              mace_pb2.DT_INT32, axis_data)
            op_.input.extend([axis_tensor_name])
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_BatchNorm(self, op):
        # yi = xi * scale_value + offset_value
        # output_shape == input_shape
        name = op.name
        xi = op.input[0]
        scale_value = op.input[1]
        offset_value = op.input[2]
        outputName = op.output[0]
        input_shape = self.get_shape_by_name(xi)
        inputTensor = self.find_tensor_by_name(xi)
        add_transpose = False
        new_shape = copy.deepcopy(input_shape)
        #mace_check(len(input_shape) <= 4, "batch norm do not support more than 4 dim input .")
        if inputTensor.data_format == mace_pb2.DT_NHWC and len(input_shape) == 4:
            new_input_name = self.check_NCHW_And_Change(xi, name)
            xi = new_input_name
            self.set_output_datatype_by_input (op, mace_pb2.DT_NCHW)
        #  input shape is N,C,D1,D2,...
        #  change to N,D1,D2,...,C
        elif len(input_shape) != 4:
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
          op_transpose.name = name + '_transpose'
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
            op_add.output[:] = op.output[:]
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
            op_transpose.name = name + '_transpose#2'
            op_transpose.type = 'TRANSPOSE'
            shape_tensor_name = op_transpose.name + '_shape'
            shape_tensor_data = out_shape_order
            shape_tensor_shape = [len(out_shape_order)]
            self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                mace_pb2.DT_INT32, shape_tensor_data)
            op_transpose.input.extend([output_name_add])
            op_transpose.input.extend([shape_tensor_name])
            op_transpose.output[:] = op.output[:]
            op_transpose.output_shape.extend(op.output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_transpose)

        #remove BatchNorm op
        self.remove_op_with_name(name)

    def split_Cast(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CAST"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Concat(self, op):
        all_input_NHWC = len(op.input)
        for j,bottom in enumerate(op.input):
            bottom_tensor = self.find_tensor_by_name(bottom)
            if bottom_tensor.data_format == mace_pb2.DT_NHWC:
                all_input_NHWC -= 1
        if all_input_NHWC == 0: #all input tensor is NHWC
            for op_ in self._SGSModel.op:
                if op_ == op:
                    inputShape = self.get_shape_by_name(op_.input[0])
                    op_.type = "CONCATENATION"
                    arg = op_.arg
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            #only convert negative axis to positive.
                            arg[i].i = self.get_axes_by_shape (arg[i].i, len(inputShape), False, True)
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    outputName = op_.output[0]
                    outputTensor = self.find_tensor_by_name(outputName)
                    outputTensor.data_format = mace_pb2.DT_NHWC
        else:
            for op_ in self._SGSModel.op:
                if op_ == op:
                    axis = 1
                    inputShape = self.get_shape_by_name(op_.input[0])
                    output_shape = op_.output_shape[:]
                    op_.type = "CONCATENATION"
                    arg = op_.arg
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            arg[i].i = self.get_axes_by_shape (arg[i].i, len(inputShape))
                    for j,bottom in enumerate(op.input):
                        bottom_tensor = self.find_tensor_by_name(bottom)
                        bottom_shape = self.get_shape_by_name(bottom)
                        if bottom_tensor.data_format == mace_pb2.DT_NHWC and len(bottom_shape) == 4:
                            new_input_name = self.check_NCHW_And_Change(bottom, op_.name)
                            op_.input[j] = new_input_name
                            self.set_output_datatype_by_input (op_, mace_pb2.DT_NCHW)
                    self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Conv1D(self, op):
        op_name = op.name
        if "\'" in op_name:
            op_name = op_name[2:-2]
        x = op.input[0]
        shape = self.get_shape_by_name(x)
        w = self.find_tensor_by_name(op.input[1]).float_data

        # create  transpose op
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = op_name + '#transpose'
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
        op_transpose.output_shape[0].dims.extend([shape[0], shape[2], shape[1]])
        self._maceOpArray = np.append(self._maceOpArray, op_transpose)

        # creat reshape op
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '#reshape'
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
        op_reshape.output_shape[0].dims.extend([op_transpose.output_shape[0].dims[0], op_transpose.output_shape[0].dims[2],
                                                1, op_transpose.output_shape[0].dims[1]])  # nchw
        self._maceOpArray = np.append(self._maceOpArray, op_reshape)

        # creat conv op
        arg = op.arg
        weight = op.input[1]
        bias = op.input[2]
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == MaceKeyword.mace_padding_values_str:
                # padding value should be divided by 2
                paddingL, paddingR, paddingT, paddingB = arg[i].ints
            elif name == MaceKeyword.mace_strides_str:
                strideH, strideW = arg[i].ints
        op_conv = self._SGSModel.op.add()
        op_conv.name = op_name + '_conv#1'
        op_conv.type = 'CONV_2D'
        #add padding_type
        paddingType_arg = op_conv.arg.add()
        paddingType_arg.name = MaceKeyword.mace_padding_str
        #set padding_type
        paddingType_arg.i = self.pooling_paddingType['CAFFE']
        strides_arg = op_conv.arg.add()
        strides_arg.name = 'strides'
        strides_arg.ints.extend([strideH, strideW])
        padding_arg = op_conv.arg.add()
        padding_arg.name = 'padding_values'
        padding_arg.ints.extend([paddingL, paddingR, paddingT, paddingB])
        op_conv.input.extend([output_op_reshape])
        op_conv.input.extend([weight])
        op_conv.input.extend([bias])
        output_op_conv = op_conv.name + '_output'
        op_conv.output.extend([output_op_conv])
        op_conv.output_shape.add()
        op_conv.output_shape[0].dims.extend([op.output_shape[0].dims[0], op.output_shape[0].dims[1],
                                             1, op.output_shape[0].dims[2]]) #nchw
        self._maceOpArray = np.append(self._maceOpArray, op_conv)

        # creat reshape op
        op_reshape2 = self._SGSModel.op.add()
        op_reshape2.name = op_name + '_reshape2'
        op_reshape2.type = 'RESHAPE'
        # op_reshape2.input.extend([output_op_conv])
        # set reshape's output_shape
        reshape2_output_tensor_name = op_reshape2.name + '_output_shape'
        reshape2_output_tensor_data = [op_conv.output_shape[0].dims[0], op_conv.output_shape[0].dims[3],
                                       op_conv.output_shape[0].dims[1]]
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
        op_reshape2.output_shape[0].dims.extend([op_conv.output_shape[0].dims[0], op_conv.output_shape[0].dims[3],
                                                 op_conv.output_shape[0].dims[1]])  # nchw
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
        op_transpose2.output_shape[0].dims.extend([op_reshape2.output_shape[0].dims[0], op_reshape2.output_shape[0].dims[2],
                                                   op_reshape2.output_shape[0].dims[1]])
        self._maceOpArray = np.append(self._maceOpArray, op_transpose2)

        #self.remove_op_with_name(op_name)

    def split_Conv2D(self, op):
       for op_ in self._SGSModel.op:
          if op_ == op:
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
              elif name == "group":
                group = arg[i].i

            inputTensor = self.find_tensor_by_name(xi)
            if inputTensor.data_format == mace_pb2.DT_NHWC:
              new_input_name = self.check_NCHW_And_Change(xi, op_name)
              op_.input[0] = new_input_name
              self.set_output_datatype_by_input (op_, mace_pb2.DT_NCHW)
            if group == 1:
                op_.type = "CONV_2D"
                #add padding_type
                paddingType_arg = arg.add()
                paddingType_arg.name = MaceKeyword.mace_padding_str
                #set padding_type
                paddingType_arg.i = self.pooling_paddingType['CAFFE']
                self._maceOpArray = np.append(self._maceOpArray,op_)

            else:
                input_shape = self.get_shape_by_name(xi)
                weight = op_.input[1]
                bias = op_.input[2]
                op_output_shape = op_.output_shape[0].dims
                tensor_weight = self.find_tensor_by_name(weight)
                tensor_weight_shape = tensor_weight.dims
                tensor_bias = self.find_tensor_by_name(bias)
                tensor_bias_shape = tensor_bias.dims
                mace_check(tensor_weight_shape[0] % group == 0,
                               "tensor weight shape 0 must be multiples of group.")
                mace_check(input_shape[1] % group == 0,
                               "tensor weight shape 0 must be multiples of group.")

                #split input according to group num
                # creat split
                op_split = self._SGSModel.op.add()
                op_split.name = op_name + '_split'
                op_split.type = 'SPLIT'
                axis_tensor_name = op_split.name + '_axis'
                axis_data = [3] # 1*224*224*3 [0,1,2,3] ->[N,H,W,C],split axis[3]->split[c]
                axis_shape = [1]
                self.add_tensor(self._SGSModel, axis_tensor_name, axis_shape,
                              mace_pb2.DT_INT32, axis_data)
                split_output_shape = copy.deepcopy(input_shape)
                split_output_shape[1] = int(input_shape[1] / group)
                op_split.input.extend([axis_tensor_name])
                op_split.input.extend([xi])
                for i in six.moves.range(group):
                  split_topName = op_split.name + '_output#' + str(i)
                  op_split.output.extend([split_topName])
                  op_split.output_shape.add()
                  op_split.output_shape[i].dims.extend(split_output_shape)
                numSplits_arg = op_split.arg.add()
                numSplits_arg.name = MaceKeyword.mace_num_split_str
                numSplits_arg.i = len(op_split.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_split)

                #creat conv according to gruop num
                output_op_conv_name_array = np.array([])
                for i in six.moves.range(group):
                    op_conv = self._SGSModel.op.add()
                    op_conv.name = op_name + '_conv#' + str(i)
                    op_conv.type = 'CONV_2D'
                    strides_arg = op_conv.arg.add()
                    strides_arg.name = 'strides'
                    strides_arg.ints.extend([strideH,strideW])
                    padding_arg = op_conv.arg.add()
                    padding_arg.name = 'padding_values'
                    padding_arg.ints.extend([paddingL,paddingR,paddingT,paddingB])
                    #add padding_type
                    paddingType_arg = op_conv.arg.add()
                    paddingType_arg.name = MaceKeyword.mace_padding_str
                    #set padding_type
                    paddingType_arg.i = self.pooling_paddingType['CAFFE']

                     #creat filter
                    filter_tensor_name = op_conv.name + "_filter#" + str(i)
                    filter_tensor_shape = copy.deepcopy(tensor_weight_shape)
                    filter_tensor_shape[0] = int(tensor_weight_shape[0] / group)
                    offset = len(tensor_weight.float_data) / group
                    filter_tesnor_value = tensor_weight.float_data[int(i*offset):int((i+1)*offset)]
                    self.add_tensor(self._SGSModel ,filter_tensor_name, filter_tensor_shape,
                            mace_pb2.DT_FLOAT, filter_tesnor_value)
                    #creat bias
                    bias_tensor_name = op_conv.name + "_bias#" + str(i)
                    bias_tensor_shape = copy.deepcopy(tensor_bias_shape)
                    bias_tensor_shape[0] = int(tensor_bias_shape[0] / group)
                    offset = len(tensor_bias.float_data) / group
                    bias_tesnor_value = tensor_bias.float_data[int(i*offset):int((i+1)*offset)]
                    self.add_tensor(self._SGSModel ,bias_tensor_name, bias_tensor_shape,
                            mace_pb2.DT_FLOAT, bias_tesnor_value)

                    op_conv.input.extend([op_split.name + '_output#' + str(i)])
                    op_conv.input.extend([filter_tensor_name])
                    op_conv.input.extend([bias_tensor_name])
                    output_op_conv_name = op_conv.name + '_output#' + str(i)
                    output_op_conv_name_array = np.append(output_op_conv_name_array,output_op_conv_name)
                    op_conv.output.extend([output_op_conv_name])
                    op_conv.output_shape.add()
                    conv_output_shape = copy.deepcopy(op_output_shape)
                    conv_output_shape[1] = int(op_output_shape[1] / group)
                    op_conv.output_shape[0].dims.extend(conv_output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_conv)

                #creat concat
                op_concat = self._SGSModel.op.add()
                op_concat.name = op_name + '_concat'
                op_concat.type = 'CONCATENATION'
                axis_arg = op_concat.arg.add()
                axis_arg.name = 'axis'
                axis_arg.i = 3
                for name in output_op_conv_name_array:
                  op_concat.input.extend([name])
                output_op_concat = op_concat.name + '_output'
                op_concat.output[:] = op.output[:]
                op_concat.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_concat)

                del tensor_weight
                del tensor_bias
                self.remove_op_with_name(op_name)

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
                  if weightTensor.int32_data != []:
                      weight_ori_data = weightTensor.int32_data
                  else:
                      weight_ori_data = weightTensor.float_data
                  if len(weight_ori_data) != 0 and len(weightShape) == 5:
                      #utility.annPrint("Reshape ",weightTensor.name,"to DHWNC")
                      data = np.array(weight_ori_data)
                      data = data.reshape(weightShape)
                      data = data.transpose(2,3,4,1,0)
                      data = list(data.flat)
                      weight_new_data = copy.deepcopy(data)
                      tmp_dim = [2,3,4,1,0]
                      tmp_shape = [0,1,2,3,4]
                      for i in six.moves.range(len(tmp_dim)):
                          tmp_shape[i] = weightShape[tmp_dim[i]]
                      weightShape_new = tmp_shape
                      weightTensor_new = weightTensor
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
                                input_tensor_shape[1] = inputShape[4]
                                input_tensor_shape[2] = inputShape[2]
                                input_tensor_shape[3] = inputShape[3]
                                # NHWC will be transposed

                                self.add_tensor(self._SGSModel ,input_tensor_name, input_tensor_shape,
                                                mace_pb2.DT_FLOAT, None)
                                input_tensor_name_array.extend([input_tensor_name])
                                self._inputName.extend([input_tensor_name])
                            self._inputName.remove(self._inputName[0])
                            inputShapeSliced = True
                            new_ini = self.CreateNewIniFor3D(INPUT_CONFIG_INI,self._inputName)
                        break

                # create transpose1->input transpose
                if inputShapeChanged == False:
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = op_name + '_transpose1'
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
        # (mul * -1 + leakRelu)    leakRelu
        #           |              |
        #           | - - - -|- - -|
        #                    |
        #                  concat
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
        if len(op.output_shape[0].dims) == 4:#nchw -> nhwc
           if axis == 1:
             axis = 3
           elif axis == 2:
             axis = 1
           elif axis == 3:
             axis = 2
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

    def split_Deconv2D(self, op):
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
        if inputTensor.data_format == mace_pb2.DT_NHWC:
            new_input_name = self.check_NCHW_And_Change(xi, op_name)
            xi = new_input_name
            self.set_output_datatype_by_input (op, mace_pb2.DT_NCHW)

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

        pad_h,pad_w,dilation_output_h,dilation_output_w = computeOutSidePadding(input_shape,output_shape,strideH,strideW,filter_data.shape[2:])
        #                #
        # create dilation #
        #                #
        op_dilation = self._SGSModel.op.add()
        op_dilation.name = 'Dilation'+ op_name
        op_dilation.type = 'CUSTOM'
        # 1.add outside pad tensor
        outside_pad_name = op.name + '_dilation' + '_outside_pad'
        outside_pad_data = [0,0,pad_h,pad_h,pad_w,pad_w,0,0] #[[n],[h],[w],[c]]
        outside_pad_shape = [4,2]
        self.add_tensor(self._SGSModel ,outside_pad_name, outside_pad_shape,
            mace_pb2.DT_INT32, outside_pad_data)
        # 2.add inside pad tensor
        inside_pad_name = op.name + '_dilation' + '_inside_pad'
        inside_pad_data = [0,int(strideH-1),int(strideW-1),0]
        inside_pad_shape = [4]
        self.add_tensor(self._SGSModel ,inside_pad_name, inside_pad_shape,
            mace_pb2.DT_INT32, inside_pad_data)
        op_dilation.input.extend([xi])
        op_dilation.input.extend([outside_pad_name])
        op_dilation.input.extend([inside_pad_name])
        output_name_dilation = op.name + '_dilation' + '_output'
        op_dilation.output.extend([output_name_dilation])
        op_dilation.output_shape.add()
        op_dilation.output_shape[0].dims.extend([ni,ci,dilation_output_h,dilation_output_w])
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
            #depth_filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(1,di,kh,kw)
            #filter_tensor.float_data[:] = depth_filter_data.flat
            #filter_tensor.dims[:] = [1,di,kh,kw]
            op_dep_conv.input.extend([filter_name])
            op_dep_conv.input.extend([biase_name])
            op_dep_conv.output[:] = op.output[:]
            op_dep_conv.output_shape.extend(op.output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_dep_conv)
        elif groupConv:
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
            filter_tensor.float_data[:] = depth_filter_data.flat
            filter_tensor.dims[:] = [ck*group,nk//group,hk,wk]
            op_conv.input.extend([output_name_dilation])
            op_conv.input.extend([filter_name])
            op_conv.input.extend([biase_name])
            op_conv.output[:] = op.output[:]
            op_conv.output_shape.extend(op.output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_conv)
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
            op_conv.input.extend([filter_name])
            op_conv.input.extend([biase_name])
            op_conv.output[:] = op.output[:]
            op_conv.output_shape.extend(op.output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_conv)
        self.remove_op_with_name(op_name)

    def split_DepthwiseConv1d(self, op):
        op_name = op.name
        if "\'" in op_name:
            op_name = op_name[2:-2]
        x = op.input[0]
        shape = self.get_shape_by_name(x)
        w = self.find_tensor_by_name(op.input[1]).float_data

        # create  transpose op
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = op_name + '#transpose'
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
        op_transpose.output_shape[0].dims.extend([shape[0], shape[2], shape[1]])
        self._maceOpArray = np.append(self._maceOpArray, op_transpose)

        # creat reshape op
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '#reshape'
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
        op_reshape.output_shape[0].dims.extend([op_transpose.output_shape[0].dims[0], op_transpose.output_shape[0].dims[2],
                                                1, op_transpose.output_shape[0].dims[1]])  # nchw
        self._maceOpArray = np.append(self._maceOpArray, op_reshape)

        # create conv op
        arg = op.arg
        weight = op.input[1]
        bias = op.input[2]
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == MaceKeyword.mace_padding_values_str:
                # padding value should be divided by 2
                paddingL, paddingR, paddingT, paddingB = arg[i].ints
            elif name == MaceKeyword.mace_strides_str:
                strideH, strideW = arg[i].ints
        op_conv = self._SGSModel.op.add()
        op_conv.name = op_name + '_conv#1'
        op_conv.type = 'DEPTHWISE_CONV_2D'
        strides_arg = op_conv.arg.add()
        strides_arg.name = 'strides'
        strides_arg.ints.extend([strideH, strideW])
        padding_arg = op_conv.arg.add()
        padding_arg.name = 'padding_values'
        padding_arg.ints.extend([paddingL, paddingR, paddingT, paddingB])
        #add padding_type
        paddingType_arg = op_conv.arg.add()
        paddingType_arg.name = MaceKeyword.mace_padding_str
        #set padding_type
        paddingType_arg.i = self.pooling_paddingType['CAFFE']
        op_conv.input.extend([output_op_reshape])
        op_conv.input.extend([weight])
        op_conv.input.extend([bias])
        output_op_conv = op_conv.name + '_output'
        op_conv.output.extend([output_op_conv])
        op_conv.output_shape.add()
        op_conv.output_shape[0].dims.extend([op_reshape.output_shape[0].dims[0], op_reshape.output_shape[0].dims[1],
                                             op_reshape.output_shape[0].dims[2], op_reshape.output_shape[0].dims[3]])  # nchw
        self._maceOpArray = np.append(self._maceOpArray, op_conv)

        # creat reshape op
        op_reshape2 = self._SGSModel.op.add()
        op_reshape2.name = op_name + '_reshape2'
        op_reshape2.type = 'RESHAPE'
        # op_reshape2.input.extend([output_op_conv])
        # set reshape's output_shape
        reshape2_output_tensor_name = op_reshape2.name + '_output_shape'
        reshape2_output_tensor_data = [op_conv.output_shape[0].dims[0], op_conv.output_shape[0].dims[3],
                                       op_conv.output_shape[0].dims[1]]
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
        op_reshape2.output_shape[0].dims.extend([op_conv.output_shape[0].dims[0], op_conv.output_shape[0].dims[3],
                                                 op_conv.output_shape[0].dims[1]])  # nchw
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
        op_transpose2.output_shape[0].dims.extend([op_reshape2.output_shape[0].dims[0], op_reshape2.output_shape[0].dims[2],
                                                   op_reshape2.output_shape[0].dims[1]])
        self._maceOpArray = np.append(self._maceOpArray, op_transpose2)

        #self.remove_op_with_name(op_name)

    def split_DepthwiseConv2d(self, op):
      for op_ in self._SGSModel.op:
          if op_ == op:
            op_name = op_.name
            input_data_name = op_.input[0]
            filter_name = op_.input[1]
            biase_name = op_.input[2]
            filter_tensor = self.find_tensor_by_name(op_.input[1])
            [n,c,h,w] = filter_tensor.dims[:]
            if filter_tensor.float_data == []:
                #weight tensor is virable tensor
                #creat transpose
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = op_name + '_transpose'
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
                op_transpose.output_shape[0].dims.extend([c,n,h,w])
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                # replace op inputs
                op_.input[1] = output_op_transpose
            # check input format
            inputTensor = self.find_tensor_by_name(op_.input[0])
            if inputTensor.data_format == mace_pb2.DT_NHWC:
              new_input_name = self.check_NCHW_And_Change(input_data_name, op_name)
              op_.input[0] = new_input_name
              self.set_output_datatype_by_input (op_, mace_pb2.DT_NCHW)

            op_.type = "DEPTHWISE_CONV_2D"
            #add padding_type
            paddingType_arg = op_.arg.add()
            paddingType_arg.name = MaceKeyword.mace_padding_str
            #set padding_type
            paddingType_arg.i = self.pooling_paddingType['CAFFE']
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Dropout(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            arg = op_.arg
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == 'dropout_ratio':
                dropout_ratio = arg[i].f
              if name == 'scale_train':
                scale_train = arg[i].i

        name = op.name
        xi = op.input[0]
        scale_value = 1 if scale_train else (1-dropout_ratio)
        #creat Mul
        op_mul = self._SGSModel.op.add()
        op_mul.name = name + '_MUL'
        op_mul.type = 'MUL'
        op_mul.input.extend([xi])
        op_mul.input.extend([scale_value])

        op_mul.output[:] = op.output[:]
        op_mul.output_shape.extend(op.output_shape)
        #tensor data is variable,so didn't creat new tensor
        self._maceOpArray = np.append(self._maceOpArray,op_mul)

    def split_DepthToSpace(self, op):
        xi = op.input[0]
        arg = op.arg
        mode_type = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'block_size':
            block_size = arg[i].i
          if name == 'mode':
            mode_type = arg[i].i

        op_name = op.name
        [n,c,h,w] = self.get_shape_by_name(xi)
        c1 = int(c/(block_size*block_size))
        w1 = w//block_size
        h1 = h//block_size

        #creat transpose
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = op_name + '_transpose'
        op_transpose.type = 'TRANSPOSE'
        shape_tensor_name = op_transpose.name + '_shape'
        shape_tensor_data = [0,3,1,2]
        shape_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
            mace_pb2.DT_INT32, shape_tensor_data)
        op_transpose.input.extend([xi])
        op_transpose.input.extend([shape_tensor_name])
        output_op_transpose = op_transpose.name + '_output'
        op_transpose.output.extend([output_op_transpose])
        op_transpose.output_shape.add()
        op_transpose.output_shape[0].dims.extend([n,w,c,h])
        self._maceOpArray = np.append(self._maceOpArray,op_transpose)

        #add reshape
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '_reshape'
        op_reshape.type = 'RESHAPE'
        op_reshape.input.extend([output_op_transpose])
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
        op_transpose.name = op_name + '_transpose#1'
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
        reshape_output_tensor_data = [n, h1, w1, c1]
        reshape_output_tensor_shape = [len(reshape_output_tensor_data)]
        self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
        op_reshape.input.extend([reshape_output_tensor_name])
        op_reshape.output[:] =  op.output[:]
        op_reshape.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_reshape)

    def split_Expand(self, op):
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
            if inputTensor.data_format == mace_pb2.DT_NCHW and len(input_shape) == 4:
                Transformer.transpose_shape(scale, [0, 2, 3, 1])
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

            if inputTensor.data_format == mace_pb2.DT_NHWC:
                for outputName in op.output:
                    outputTensor = self.find_tensor_by_name(outputName)
                    outputTensor.data_format = mace_pb2.DT_NHWC
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
                if inputTensor.data_format == mace_pb2.DT_NCHW and len(output_shape) == 4:
                    Transformer.transpose_shape(scale, [0, 2, 3, 1])
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
        # Calculation Description
        # case1: Both operands are 4-dimensional
        #    support [1,3,2,4] × [1,3,2,4], Need to unify data layout
        #    support [1,3,2,4] × [1,3,1,1], C dimension needs to be in the innermost dimension
        #    support [1,3,1,1] × [1,3,2,4], C dimension needs to be in the innermost dimension
        #    support [1,3,1,1] × [1,3,1,1], C dimension needs to be in the innermost dimension
        #    support [1,3,2,4] × [1,1,2,1]
        #    support [1,3,2,4] × [1,1,1,4]
        #
        # case2: One operand dimension is 4, another operand dimension is 3
        #    support [3,1,1] × [1,3,2,4], Need to multiply vector [3,1,1] into [3],
        #                                 C dimension needs to be in the innermost dimension
        #    *unsupport [1,2,1] × [1,3,2,4],
        #    *unsupport [1,1,4] × [1,3,2,4],
        #
        # case3: One operand dimension is 4, another operand dimension is 2
        #    *unsupport [2,1] × [1,3,2,4],
        #    *unsupport [1,4] × [1,3,2,4],
        #
        # case4: One operand dimension is 4, another operand dimension is 1
        # C dimension needs to be in the innermost dimension:
        #    support [1,3,2,4] × [4], C dimension needs to be in the innermost dimension
        #    support [1,3,2,4] × [1], C dimension needs to be in the innermost dimension
        #
        # case5: none of input is 4dim: do nothing

        # case 1
        def _split_Eltwise_case1(op_, x1, x2, inputTensor1, inputTensor2 ,inputShape1, inputShape2, op_type, change_format_enable = True):
            op_name = op_.name
            if (inputTensor1.data_format != inputTensor2.data_format) and op_type != 'POW':
                if inputTensor1.data_format == mace_pb2.DT_NHWC and self.is_scalar_tensor(x2) == False \
                    or inputTensor2.data_format == mace_pb2.DT_NHWC and self.is_scalar_tensor(x1) == False:
                    change_index = 0 if inputTensor1.data_format == mace_pb2.DT_NHWC else 1
                    new_input_name = self.check_NCHW_And_Change(op_.input[change_index], op_name)
                    op_.input[change_index] = new_input_name
                    for outputName in op_.output:
                        outputTensor = self.find_tensor_by_name(outputName)
                        outputTensor.data_format = mace_pb2.DT_NCHW
                    change_format_enable = False
            if op_type == 'POW':
                return self.split_Pow(op_, change_format_enable)
            self._maceOpArray = np.append(self._maceOpArray, op_)
            if change_format_enable:
                if (inputTensor1.data_format == mace_pb2.DT_NHWC and inputTensor2.data_format == mace_pb2.DT_NHWC) or  \
                        (inputTensor1.data_format == mace_pb2.DT_NHWC and inputTensor2.data_format == mace_pb2.DT_NCHW and len(inputTensor1.dims)==4) or \
                        (inputTensor1.data_format == mace_pb2.DT_NCHW and inputTensor2.data_format == mace_pb2.DT_NHWC and len(inputTensor2.dims)==4):
                    for outputName in op_.output:
                        outputTensor = self.find_tensor_by_name(outputName)
                        outputTensor.data_format = mace_pb2.DT_NHWC
        # case 2
        def _split_Eltwise_case2(op_, x1, x2, inputTensor1, inputTensor2 ,inputShape1, inputShape2, op_type, change_format_enable = True):
            op_name = op_.name
            for inputName in op_.input:
                shape_total = 1
                if self.is_const_tensor(inputName) == True:
                    input_shape = self.get_shape_by_name(inputName)
                    if len(input_shape) != 4:
                        for shape in input_shape:
                            shape_total *= shape
                        if shape_total in input_shape:
                            inputTensor = self.find_tensor_by_name(inputName)
                            for i in six.moves.range(len(input_shape)):
                                inputTensor.dims.pop()
                            inputTensor.dims.extend([shape_total])
                        else:
                            mace_check(shape_total in input_shape, "Does not support type %s" % op_.type)
            # Need to unify data layout,
            # if one operator will be changed to TF, need to change another tensor which won't be changed,
            if (inputTensor1.data_format != inputTensor2.data_format) and op_type != 'POW':
                if inputTensor1.data_format == mace_pb2.DT_NHWC and self.is_scalar_tensor(x2) == False \
                    or inputTensor2.data_format == mace_pb2.DT_NHWC and self.is_scalar_tensor(x1) == False:
                    change_index = 0 if inputTensor1.data_format == mace_pb2.DT_NHWC else 1
                    new_input_name = self.check_NCHW_And_Change(op_.input[change_index], op_name)
                    op_.input[change_index] = new_input_name
                    for outputName in op_.output:
                        outputTensor = self.find_tensor_by_name(outputName)
                        outputTensor.data_format = mace_pb2.DT_NCHW
                    change_format_enable = False
            if op_type == 'POW':
                return self.split_Pow(op_, change_format_enable)
            self._maceOpArray = np.append(self._maceOpArray, op_)
            if change_format_enable:
                if (inputTensor1.data_format == mace_pb2.DT_NHWC and inputTensor2.data_format == mace_pb2.DT_NHWC) or  \
                        (inputTensor1.data_format == mace_pb2.DT_NHWC and inputTensor2.data_format == mace_pb2.DT_NCHW and len(inputTensor1.dims)==4) or \
                        (inputTensor1.data_format == mace_pb2.DT_NCHW and inputTensor2.data_format == mace_pb2.DT_NHWC and len(inputTensor2.dims)==4):
                    for outputName in op_.output:
                        outputTensor = self.find_tensor_by_name(outputName)
                        outputTensor.data_format = mace_pb2.DT_NHWC
        # case 3
        def _split_Eltwise_case3(op_, x1, x2, inputTensor1, inputTensor2 ,inputShape1, inputShape2, op_type, outputTensor, output_shape, change_format_enable = True, inputChanged = False):
            op_name = op.name
            if (len(inputShape1) == 4 and len(inputShape2) == 1 and inputShape1[3] == inputShape2[0] != 1 and self.is_scalar_tensor(x2) == False and op_.type != 'POW') \
                or (len(inputShape1) == 4 and len(inputShape2) == 1 and inputShape2[0] == 1 and self.is_scalar_tensor(x2) == False and op_.type != 'POW')  :
                # change back to ONNX which will be changed to TF
                if inputTensor1.data_format == mace_pb2.DT_NCHW:
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = op_name +'_input0_transpose'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = [0,3,1,2]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                    mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([x1])
                    op_transpose.input.extend([shape_tensor_name])
                    output_op_transpose = op_transpose.name + '_output'
                    op_transpose.output.extend([output_op_transpose])
                    tmp_dim = [0,3,1,2]
                    tmp_shape = [0,1,2,3]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = inputShape1[tmp_dim[i]]
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(tmp_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    op_.input[0] = output_op_transpose
                    # del inputTensor1
                    inputChanged = True

            elif (len(inputShape2) == 4 and len(inputShape1) == 1 and inputShape2[3] == inputShape1[0] != 1 and self.is_scalar_tensor(x1) == False and op_.type != 'POW') \
                or (len(inputShape2) == 4 and len(inputShape1) == 1 and inputShape1[0] == 1 and self.is_scalar_tensor(x1) == False and op_.type != 'POW'):
                # change back to ONNX which will be changed to TF
                if inputTensor2.data_format == mace_pb2.DT_NCHW:
                    if self.is_const_tensor(inputTensor2) == False:
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = op_name +'_input1_transpose'
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shape'
                        shape_tensor_data = [0,3,1,2]
                        shape_tensor_shape = [4]
                        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                        mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([x2])
                        op_transpose.input.extend([shape_tensor_name])
                        output_op_transpose = op_transpose.name + '_output'
                        op_transpose.output.extend([output_op_transpose])
                        tmp_dim = [0,3,1,2]
                        tmp_shape = [0,1,2,3]
                        for i in six.moves.range(len(tmp_dim)):
                            tmp_shape[i] = inputShape2[tmp_dim[i]]
                        op_transpose.output_shape.add()
                        op_transpose_shape = copy.deepcopy(tmp_shape)
                        op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                        op_.input[1] = output_op_transpose
                        # del inputTensor2
                        inputChanged = True

                    elif self.is_const_tensor(inputTensor2) == True:
                        if self.find_tensor_by_name(inputTensor2).int32_data != []:
                            const_input_ori_data = self.find_tensor_by_name(inputTensor2).int32_data
                        else:
                            const_input_ori_data = self.find_tensor_by_name(inputTensor2).float_data
                        if len(const_input_ori_data) != 0 and len(inputShape2) == 4:
                            data = np.array(const_input_ori_data)
                            data = data.reshape(inputShape2)
                            data = data.transpose(0,3,1,2)
                            data = list(data.flat)
                            new_data = copy.deepcopy(data)
                            tmp_dim = [0,3,1,2]
                            tmp_shape = [0,1,2,3]
                            for i in six.moves.range(len(tmp_dim)):
                                tmp_shape[i] = inputShape2[tmp_dim[i]]
                            for i in six.moves.range(len(tmp_dim)):
                                inputShape2[i] = tmp_shape[i]
                            const_tensor_name = op_name + '_constB'
                            const_tensor_shape = inputShape2
                            const_tensor_data = data
                            self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                            mace_pb2.DT_FLOAT, const_tensor_data)
                            op_.input[1] = const_tensor_name
                            #del inputTensor2
                            inputChanged = True
            else:
                mace_check(op_.type == 'POW' or self.is_scalar_tensor(x2) == True or self.is_scalar_tensor(x1) == True, "Does not support elt type %s" % op_.type)

            if inputChanged == True:
                output_op_elt = op_name + '_elt_output'
                self.create_Eltwise_op(op_, op_.type, output_op_elt, output_shape)
                # add transpose for output
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = op_name + '_transposeOut'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shapeOut'
                shape_tensor_data = [0,2,3,1]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([output_op_elt])
                op_transpose.input.extend([shape_tensor_name])
                # transpose input
                tmp_dim = [0,2,3,1]
                tmp_shape = [0,1,2,3]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = output_shape[tmp_dim[i]]
                op_transpose.output.extend([outputTensor])
                op_transpose.output_shape.add()
                op_transpose_shape = copy.deepcopy(tmp_shape)
                op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                # self.add_tensor(self._SGSModel, C, op_transpose_shape, mace_pb2.DT_FLOAT,
                #                 None, data_fromat = mace_pb2.DT_NHWC)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                self.remove_op_with_name(op_name)
            else:
                if op_.type == 'POW':
                    return self.split_Pow(op_, change_format_enable)
                self._maceOpArray = np.append(self._maceOpArray, op_)

            if len(op_.input) > 1 and change_format_enable:
                if (inputTensor1.data_format == mace_pb2.DT_NHWC and inputTensor2.data_format == mace_pb2.DT_NHWC) or  \
                        (inputTensor1.data_format == mace_pb2.DT_NHWC and inputTensor2.data_format == mace_pb2.DT_NCHW and len(inputTensor1.dims)==4) or \
                        (inputTensor1.data_format == mace_pb2.DT_NCHW and inputTensor2.data_format == mace_pb2.DT_NHWC and len(inputTensor2.dims)==4):
                    for outputName in op_.output:
                        outputTensor = self.find_tensor_by_name(outputName)
                        outputTensor.data_format = mace_pb2.DT_NHWC

        # main
        for op_ in self._SGSModel.op:
            if op_ == op:
                # 1.get op arg
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
                        elif type == 10:
                            op_.type = 'EQUAL'
                        else:
                            mace_check(False, "Does not support this eltwise op type %s" % op_name)

                # 2.set data flag
                change_format_enable = True
                # 3.get op input
                inputNum = len(op_.input)
                if inputNum > 1:
                    x1 = op_.input[0]
                    inputTensor1 = self.find_tensor_by_name(x1)
                    inputShape1 = self.get_shape_by_name(x1)
                    x2 = op_.input[1]
                    inputTensor2 = self.find_tensor_by_name(x2)
                    inputShape2 = self.get_shape_by_name(x2)
                    if self.is_scalar_tensor(x1):
                        inputShape1 = [1]
                    if self.is_scalar_tensor(x2):
                        inputShape2 = [1]
                    outputTensor = op_.output[0]
                    output_shape = self.get_shape_by_name(outputTensor)
                    # case1:
                    if len(inputShape1) == 4 and len(inputShape1) == len(inputShape2):
                        # Need to unify data layout,
                        # if one operator will be changed to TF, need to change another tensor which won't be changed,
                        _split_Eltwise_case1(op_, x1, x2, inputTensor1, inputTensor2 ,inputShape1, inputShape2, op_.type, True)
                    # case2:
                    elif (len(inputShape1) == 4 and len(inputShape2) == 3) or (len(inputShape2) == 4 and len(inputShape1) == 3):
                        # mace check support cases:
                        #  support [3,1,1] × [1,3,2,4]
                        # *unsupport [1,2,1] × [1,3,2,4],
                        # *unsupport [1,1,4] × [1,3,2,4],
                        mace_check(inputShape1[1] == inputShape2[0] or inputShape2[1] == inputShape1[0],
                                   "Does not support type %s" % op_.type)
                        # multiply vector [3,1,1] into [3]
                        _split_Eltwise_case2(op_, x1, x2, inputTensor1, inputTensor2 ,inputShape1, inputShape2, op_.type, True)
                    # case3:
                    elif (len(inputShape1) == 4 and len(inputShape2) == 1) or (len(inputShape2) == 4 and len(inputShape1) == 1):
                        # support [1,3,2,4] × [4]
                        # support [1,3,2,4] × [1]
                        _split_Eltwise_case3(op_, x1, x2, inputTensor1, inputTensor2 ,inputShape1, inputShape2, op_.type, outputTensor, output_shape, True, False)
                    # other cases:
                    else:
                        if op_.type == 'POW':
                            return self.split_Pow(op_, change_format_enable)
                        else:
                            self._maceOpArray = np.append(self._maceOpArray, op_)

                else:
                    if op_.type == 'POW':
                        return self.split_Pow(op_, change_format_enable)
                    else:
                        self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Pow(self, op, change_format_enable):
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
                    op_.type = 'SQRT'
                    # sqrt only needs input0
                    if len(op.input) == 2:
                        op_.input.pop()
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                elif power % 1 != 0:
                    mace_check(False, "POW only support non-integer exponent 0.5")
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

                if inputTensor1 != False and change_format_enable:
                    if len(op_.input) > 1 and (inputTensor1.data_format == mace_pb2.DT_NHWC or const_tensor.data_format == mace_pb2.DT_NHWC):
                        for outputName in op_.output:
                            outputTensor = self.find_tensor_by_name(outputName)
                            outputTensor.data_format = mace_pb2.DT_NHWC

    def split_Exp(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "EXP"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Where(self, op):
        def is_need_broadcast(len_dims, tensor_dims, output_dims):
            tmp_list = []
            tile_param = []
            if len_dims == 4:
                if (tensor_dims == [1, 1, 1, 1] or tensor_dims == [1,output_dims[1], 1, 1]
                        or tensor_dims == output_dims):
                    return False, None
                else:
                    for index, i in enumerate(output_shape):
                        if i != where_const_tensor.dims[index]:
                            tile_param.append(output_shape[index])
                        else:
                            tile_param.append(1)
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

        op_name = op.name
        for op_ in self._SGSModel.op:
            if op_ == op:
                output = op_.output[0]
                output_shape = self.get_shape_by_name(output)
                for outputName in op_.output:
                    outputTensor = self.find_tensor_by_name(outputName)
                    outputTensor.data_format = mace_pb2.DT_NCHW
                for index1, m in enumerate(op_.input):
                    input_tensor = self.find_tensor_by_name(m)
                    if input_tensor.data_type == mace_pb2.DT_INT32:
                        if len(input_tensor.int32_data) != 0:
                            where_const_tensor = input_tensor
                            where_name_index = index1
                            is_need, tile_param = is_need_broadcast(len(where_const_tensor.dims),
                                                        where_const_tensor.dims, output_shape)
                            if is_need:
                                where_const_ndarray = np.array(where_const_tensor.int32_data)
                                where_const_ndarray = np.reshape(where_const_ndarray, where_const_tensor.dims)
                                where_const_tensor_1 = np.tile(where_const_ndarray, tile_param)
                                where_const_tensor_1 = where_const_tensor_1.flatten()
                                where_const_tensor_1 = list(where_const_tensor_1)
                                op_where_name = op_name + '_where#' + str(index1)
                                self.add_tensor(self._SGSModel, op_where_name, output_shape,
                                                mace_pb2.DT_FLOAT, where_const_tensor_1)
                                op_.input[where_name_index] = op_where_name
                        else:
                            if len(input_tensor.dims) == 4 and input_tensor.data_format == mace_pb2.DT_NHWC:
                                new_input_name = self.check_NCHW_And_Change(op_.input[index1], op_name)
                                op_.input[index1] = new_input_name
                                self.set_output_datatype_by_input (op_, mace_pb2.DT_NCHW)
                    elif input_tensor.data_type == mace_pb2.DT_FLOAT:
                        if len(input_tensor.float_data) != 0:
                            where_const_tensor = input_tensor
                            where_name_index = index1
                            is_need, tile_param = is_need_broadcast(len(where_const_tensor.dims),
                                                        where_const_tensor.dims, output_shape)
                            if is_need:
                                where_const_ndarray = np.array(where_const_tensor.float_data)
                                where_const_ndarray = np.reshape(where_const_ndarray, where_const_tensor.dims)
                                where_const_tensor_1 = np.tile(where_const_ndarray, tile_param)
                                where_const_tensor_1 = where_const_tensor_1.flatten()
                                where_const_tensor_1 = list(where_const_tensor_1)
                                op_where_name = op_name + '_where#' + str(index1)
                                self.add_tensor(self._SGSModel, op_where_name, output_shape,
                                                mace_pb2.DT_FLOAT, where_const_tensor_1)
                                op_.input[where_name_index] = op_where_name
                        else:
                            if len(input_tensor.dims) == 4 and input_tensor.data_format == mace_pb2.DT_NHWC:
                                new_input_name = self.check_NCHW_And_Change(op_.input[index1], op_name)
                                op_.input[index1] = new_input_name
                                self.set_output_datatype_by_input(op_, mace_pb2.DT_NCHW)

                    else:
                        mace_check(False, "Does not support param's data type %s" % input_tensor.data_type)
                op_.type = "SELECT"
                self._maceOpArray = np.append(self._maceOpArray, op_)



    def split_Greater(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "GREATER"
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_GreaterOrEqual(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "GREATER_EQUAL"
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Elu(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                op_.type = "ELU"
                op_.name = 'ELU'+ op_name
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_scatternd(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                op_.type = 'CUSTOM'
                op_.name = 'Customized_ScatterND'+ op_name
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Erf(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                op_.type = "CUSTOM"
                op_.name = 'Erf' + op_name
                self._maceOpArray = np.append(self._maceOpArray, op_)


    def split_FullyConnected(self, op):
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

            if len(data_shape) == 4:
                if self.find_tensor_by_name(op_.input[0]).data_format == mace_pb2.DT_NCHW:
                    if self.is_const_tensor(op_.input[0]) == False:
                        op_transpose = self._SGSModel.op.add()
                        op_transpose.name = op_name +'_transpose'
                        op_transpose.type = 'TRANSPOSE'
                        shape_tensor_name = op_transpose.name + '_shape'
                        shape_tensor_data = [0,3,1,2]
                        shape_tensor_shape = [4]
                        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                        op_transpose.input.extend([op_.input[0]])
                        op_transpose.input.extend([shape_tensor_name])
                        output_op_transpose = op_transpose.name + '_output'
                        op_transpose.output.extend([output_op_transpose])
                        op_transpose.output_shape.add()
                        op_transpose_shape = copy.deepcopy(data_shape)
                        op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                        self.add_tensor(self._SGSModel, output_op_transpose, op_transpose_shape,
                                        mace_pb2.DT_FLOAT, None, data_fromat = mace_pb2.DT_NHWC)
                        self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                        op_.input[0] = output_op_transpose
                    else:
                        mace_check(self.is_const_tensor(op_.input[0]) == False , "do not support const tensor for input[0]")
            if len(output_shape) == 4:
                for outputName in op_.output:
                    outputTensor = self.find_tensor_by_name(outputName)
                    outputTensor.data_format = mace_pb2.DT_NHWC

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
            op_mul.output_shape[0].dims.extend(op_shape)
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
            op_add.output_shape[0].dims.extend(op_shape)
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
            op_min.output_shape[0].dims.extend(op_shape)
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
            op_relu.output_shape[0].dims.extend(op_out_shape)
            self._maceOpArray = np.append(self._maceOpArray, op_relu)
            self.remove_op_with_name(op_name)

    def split_InstanceNorm(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                input_tensor_name = op_.input[0]
                scale_tensor_name = op_.input[1]
                bias_tensor_name = op_.input[2]
                arg = op_.arg
                op_name = op_.name

                input_tensor = self.find_tensor_by_name(input_tensor_name)
                scale_tensor = self.find_tensor_by_name(scale_tensor_name)
                bias_tensor = self.find_tensor_by_name(bias_tensor_name)
                input_shape = self.get_shape_by_name(input_tensor_name)
                scale_shape = self.get_shape_by_name(scale_tensor_name)
                bias_shape = self.get_shape_by_name(bias_tensor_name)

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
                    if input_tensor.data_format == mace_pb2.DT_NHWC:
                        new_input_name = self.check_NCHW_And_Change(input_tensor_name, op_name)
                        op_.input[0] = new_input_name
                        for output_tensor_name in op_.output:
                            output_tensor = self.find_tensor_by_name(output_tensor_name)
                            output_tensor.data_format = mace_pb2.DT_NCHW
                    op_.type = 'CUSTOM'
                    op_.name = 'InstanceNorm'+ op_name
                    self._maceOpArray = np.append(self._maceOpArray,op_)
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
                    op_instancenorm.name = 'InstanceNorm'+ op_name
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
        op_name = op.name
        xi = op.input[0]
        xi_shape = self.get_shape_by_name(xi)

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

        #chang inputs from 3 dims to 4 dims to fit sgs rule
        if len(xi_shape) != 4 or xi_shape[1] !=1 or xi_shape[2] !=1:
          gru_data_input_tensor = _Add_Reshape_For_Input(op_name, xi, xi_shape)

        name_prefix = "sgs_subnet_gru" + str(self._gru_num)
        num_output = op.output_shape[0].dims[-1]
        #add indicator tensor
        t = op.output_shape[0].dims[0]
        T_time = t
        indicator_shape = [t,1]
        indicator_data = np.ones(t,dtype=np.float32).reshape(t,1)

        indicator_data[0,0] = 0.0
        indicator_tensor_name = name_prefix + '_indicator'
        self.add_tensor(self._SGSModel,indicator_tensor_name, indicator_data.shape,
                        mace_pb2.DT_FLOAT,
                        indicator_data)
        # add h0
        h0_name = name_prefix + '_h0'
        h0_data = np.zeros((1,num_output,1,xi_shape[-2]),dtype=np.float32)
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
        op_SGS_GRU.name = 'SGS_GRU'
        op_SGS_GRU.type = 'CUSTOM'

        #add inputs
        op_SGS_GRU.input.extend([gru_data_input_tensor])
        op_SGS_GRU.input.extend([indicator_tensor_name])

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
        output_op_concat = op.output[0] #norm lstm
        concat_shape = [1,1,1,1]
        concat_shape[0] = op.output_shape[0].dims[0]
        concat_shape[1] = op.output_shape[0].dims[3]
        concat_shape[3] = op.output_shape[0].dims[2]
        op_concat.output.extend([output_op_concat])
        op_concat.output_shape.add()
        op_concat.output_shape[0].dims.extend(concat_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_concat)

    def split_LSTM(self, op):
        op_name = op.name
        xi = op.input[0]
        xi_shape = self.get_shape_by_name(xi)

        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'lstm_type':
            lstm_type = arg[i].i


        lstm_input_arrays = []
        lstm_data_input_tensor = xi
        h0_input = None
        c0_input = None
        if lstm_type == 1:#h0 c0 are model inputs
          h0_input = op.input[5]
          c0_input = op.input[6]

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
          op_reshape.output_shape[0].dims.extend([input_shape[0],input_shape[2],1,input_shape[1]])#n,c,h,w
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
          lstm_data_input_tensor = _Add_Reshape_For_Input(op_name, xi, xi_shape)
        if lstm_type == 1:#h0 c0 are model inputs
          h0_shape =  self.get_shape_by_name(op.input[5])
          if len(h0_shape) != 4:
            h0_input = _Add_Reshape_For_Input(op.input[5], op.input[5], h0_shape)
          c0_shape = self.get_shape_by_name(op.input[6])
          if len(c0_shape) != 4:
            c0_input = _Add_Reshape_For_Input(op.input[6], op.input[6], c0_shape)


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

          strideSlice      |
              |            |
            lstm      +   lstm    or    coming soon
              |            |
            concat        concat
              |            |
           stridSlice      |
              |------------|
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
              op_slice.output_shape[0].dims.extend([xi_shape[0],xi_shape[2],1,xi_shape[1]])#n,c,h,w
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
            h0_data = np.zeros((1,num_output,1,xi_shape[-2]),dtype=np.float32)
            h0_shape = [1,num_output,1,xi_shape[-2]]
            self.add_tensor(self._SGSModel, h0_name, h0_shape,
                            mace_pb2.DT_FLOAT, h0_data.flatten().tolist())
            # add c0
            c0_name = name_prefix + '_c0'
            c0_data = np.zeros((1,num_output,1,xi_shape[-2]),dtype=np.float32)
            c0_shape = [1,num_output,1,xi_shape[-2]]
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
            op_SGS_LSTM.name = 'SGS_LSTM'+ op_name
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
            single_shape[1] = op.output_shape[0].dims[-1]
            single_shape[3] = xi_shape[-2]

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
            concat_shape[1] = op.output_shape[0].dims[3]
            concat_shape[3] = op.output_shape[0].dims[2]
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
              outputTensor.data_format = mace_pb2.DT_NHWC
              return output_op_concat

        #norm lstm
        if op.output_shape[0].dims[1] == 1:
          _Creat_LSTM (op, lstm_data_input_tensor, h0_input, c0_input)
        #bi lstm
        if op.output_shape[0].dims[1] == 2:

          forward_part = _Creat_LSTM (op, lstm_data_input_tensor, h0_input, c0_input, False, True)
          reverse_part = _Creat_LSTM (op, lstm_data_input_tensor, h0_input, c0_input, True, False)

          # creat total concat
          op_concat = self._SGSModel.op.add()
          op_concat.name = op_name + 'total_concat'
          op_concat.type = 'CONCATENATION'
          axis_arg = op_concat.arg.add()
          axis_arg.name = 'axis'
          axis_arg.i = 1

          op_concat.input.extend([forward_part])
          op_concat.input.extend([reverse_part])
          output_op_concat_total = op.output[0]
          concat_shape = [1,1,1,1]
          concat_shape = op.output_shape[0].dims
          op_concat.output.extend([output_op_concat_total])
          op_concat.output_shape.add()
          op_concat.output_shape[0].dims.extend(concat_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_concat)
          outputTensor = self.find_tensor_by_name(output_op_concat_total)
          outputTensor.data_format = mace_pb2.DT_NHWC


        self.remove_op_with_name(op_name)

    def split_MatMul(self, op):
        #matmul A * B = C
        #
        #case 1: A:dims 2
        #   switch 1:A:dims 2  variable tensor
        #            B:dims 2  variable tensor
        #   switch 2:A:dims 2  variable tensor
        #            B:dims 2  const tensor
        #   switch 3:A:dims 2  const tensor
        #            B:dims 2  vairable tensor
        #   switch 4:A:dims 2  variable tensor
        #            B:dims 1  const tensor
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
        #
        # TIPS: if A dims > 2,Only supports the case where the HW dimension is not 1,like [1,1,x,y] where x>1,y>1

        def _Adims2xBdims2(op, op_name, A, B, A_shape, B_shape, output, output_shape):
            A_dim_num = len(A_shape)
            B_dim_num = len(B_shape)
            input_tensor_name = A
            input_tensor = self.find_tensor_by_name(input_tensor_name)
            transA = False
            transB = False
            weights_name =  B
            weights = self.find_tensor_by_name(weights_name)
            weights_dims = weights.dims

            arg = op.arg
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_transpose_a_str:
                    transA = arg[i].i
                elif name == MaceKeyword.mace_transpose_b_str:
                    transB = arg[i].i

            # TransA must be done when it is required
            # But as we use Conv2D instead of matmul,
            # Conv input's mostInner dim should equal to tensorB's mostInner dim.
            # So we should not do transB when transB is required
            # case 0: TransA == 0 TransB == 0: Transpos   B
            # case 1: TransA == 0 TransB == 1: Transpos
            # case 2: TransA == 1 TransB == 0: Transpos A B
            # case 3: TransA == 1 TransB == 1: Transpos A
            if transA == True:
                op_transposeA = self._SGSModel.op.add()
                op_transposeA.name = op_name + '_transposeA'
                op_transposeA.type = 'TRANSPOSE'
                shape_tensor_name = op_transposeA.name + '_shape'
                shape_tensor_data = [1,0]
                shape_tensor_shape = [2]
                self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape,
                    mace_pb2.DT_INT32, shape_tensor_data)
                op_transposeA.input.extend([input_tensor_name])
                op_transposeA.input.extend([shape_tensor_name])
                output_op_transposeA = op_transposeA.name + '_output'
                op_transposeA.output.extend([output_op_transposeA])
                tmp_shape = [A_shape[1],A_shape[0]]
                op_transposeA.output_shape.add()
                op_transposeA.output_shape[0].dims.extend(tmp_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transposeA)
                input_tensor_name = output_op_transposeA

            if transB == False:
                if B_dim_num == 2:
                    if len(weights.float_data) == 0:
                        op_transposeB = self._SGSModel.op.add()
                        op_transposeB.name = op_name + '_transposeB'
                        op_transposeB.type = 'TRANSPOSE'
                        shape_tensor_name = op_transposeB.name + '_shape'
                        shape_tensor_data = [1,0]
                        shape_tensor_shape = [2]
                        self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape,
                            mace_pb2.DT_INT32, shape_tensor_data)
                        op_transposeB.input.extend([weights_name])
                        op_transposeB.input.extend([shape_tensor_name])
                        output_op_transposeB = op_transposeB.name + '_output'
                        op_transposeB.output.extend([output_op_transposeB])
                        tmp_shape = [weights_dims[1],weights_dims[0]]
                        op_transposeB.output_shape.add()
                        op_transposeB.output_shape[0].dims.extend(tmp_shape)
                        self._maceOpArray = np.append(self._maceOpArray,op_transposeB)
                        weights_name = output_op_transposeB
                    else:
                        weights_new_data = list(np.array(weights.float_data).reshape(weights_dims).transpose(1, 0).flatten())
                        weights.Clear()
                        weights.name = weights_name
                        weights.dims.extend([weights_dims[1], weights_dims[0]])
                        weights.data_format = mace_pb2.DT_FLOAT
                        weights.data_type = mace_pb2.DT_FLOAT
                        weights.float_data.extend(weights_new_data)
                elif B_dim_num == 1:
                    # example: [64,256] * [256],
                    # ——onnx will broadcast [256] to [256,1]
                    # ——mace need to reshape [256] to [256,1],but if transB == False,also need to do transpose after reshape
                    # ——Since transpose swapped with 1 is equivalent to reshape，we only need to reshape [256] into [1,256]
                    # ——deal const case, Variable is not supported yet.
                    weights_new_data = list(np.array(weights.float_data).reshape(weights_dims).flatten())
                    weights.Clear()
                    weights.name = weights_name
                    weights.dims.extend([1, weights_dims[0]])
                    weights.data_format = mace_pb2.DT_FLOAT
                    weights.data_type = mace_pb2.DT_FLOAT
                    weights.float_data.extend(weights_new_data)
            if B_dim_num == 1 and transB:
                # example: [64,256] * [256],
                # ——onnx will broadcast [256] to [256,1]
                # ——mace need to reshape [256] to [256,1],if transB == T,no need to do transpose after reshape
                # ——we only need to reshape [256] into [256,1]
                # ——deal const case, Variable is not supported yet.
                weights_new_data = list(np.array(weights.float_data).reshape(weights_dims).flatten())
                weights.Clear()
                weights.name = weights_name
                weights.dims.extend([weights_dims[0],1])
                weights.data_format = mace_pb2.DT_FLOAT
                weights.data_type = mace_pb2.DT_FLOAT
                weights.float_data.extend(weights_new_data)

            #creat fullyconnet
            op_fc = self._SGSModel.op.add()
            op_fc.name = op_name + '_fc'
            op_fc.type = 'FULLY_CONNECTED'

            if len(op.input) < 3:
                bias_data = np.zeros(output_shape[-1])
                bias_tensor_name = op_fc.name + '_bias'
                self.add_tensor(self._SGSModel,bias_tensor_name, bias_data.shape,
                                mace_pb2.DT_FLOAT,bias_data)
            else:
                bias_tensor_name = op.input[2]
            op_fc.input.extend([input_tensor_name])    # input
            op_fc.input.extend([weights_name])         # weight
            op_fc.input.extend([bias_tensor_name])     # bias
            op_fc.output.extend([output])
            op_fc.output_shape.add()
            op_fc.output_shape[0].dims.extend(output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_fc)
            return op

        def _Adims3(op, op_name, A, B, A_shape, B_shape, output, output_shape):
            A_dim_num = len(A_shape)
            B_dim_num = len(B_shape)
            mace_check(A_shape[0] == 1,"the first dimension of the left operand is greater than 1 is not supported!")
            if B_dim_num == 3:
                mace_check(B_shape[0] == 1,"the first dimension of the right operand is greater than 1 is not supported!")

            op_left_output_shape = [1,1]
            for i in six.moves.range(A_dim_num - 1):
                op_left_output_shape[0] *= A_shape[i]
            op_left_output_shape[1] = A_shape[A_dim_num - 1]
            # creat reshape to change 3 dim to 2 dim
            op_reshape = self._SGSModel.op.add()
            op_reshape.name = op_name + '_reshape2'
            op_reshape.type = 'RESHAPE'
            reshape_output_tensor_name = op_reshape.name + '_output_shape2'
            reshape_output_tensor_data = op_left_output_shape
            reshape_output_tensor_shape = [len(op_left_output_shape)]
            self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                        mace_pb2.DT_INT32, reshape_output_tensor_data)
            op_reshape.input.extend([A])
            op_reshape.input.extend([reshape_output_tensor_name])
            op_reshape_output_name = op_reshape.name + '_output2'
            op_reshape.output.extend([op_reshape_output_name])
            op_reshape.output_shape.add()
            op_reshape.output_shape[0].dims.extend(op_left_output_shape)
            self.add_tensor(self._SGSModel, op_reshape_output_name, op_left_output_shape,
                        mace_pb2.DT_FLOAT, None)
            self._maceOpArray = np.append(self._maceOpArray,op_reshape)
            A_reshape_3to2dim = op_reshape_output_name

            if B_dim_num == 2:
                # called 2dim function
                A_2dim = A_reshape_3to2dim
                B_2dim = B
                A_shape_2dim = op_left_output_shape
                B_shape_2dim = B_shape
                output_2dim = op_name + '_output_2dim'
                output_shape_2dim = [A_shape_2dim[-2],B_shape_2dim[-1]]
                _Adims2xBdims2(op, op_name, A_2dim, B_2dim, A_shape_2dim, B_shape_2dim, output_2dim, output_shape_2dim)

                # create reshape to change 2 dim to 3 dim
                reshape_output_shape = output_shape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape3'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shape3'
                reshape_output_tensor_data = reshape_output_shape
                reshape_output_tensor_shape = [len(reshape_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                            mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([output_2dim])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_output3'
                op_reshape.output.extend([output])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
            elif B_dim_num == 3 and self.is_const_tensor(B) == False:
                op_right_output_shape = [B_shape[1],B_shape[2]]
                # creat reshape to change 3 dim to 2 dim
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshapeb'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shapeb'
                reshape_output_tensor_data = op_right_output_shape
                reshape_output_tensor_shape = [len(op_right_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                            mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([B])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_outputb'
                op_reshape.output.extend([op_reshape_output_name])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(op_right_output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                self.add_tensor(self._SGSModel, op_reshape_output_name, op_right_output_shape,
                            mace_pb2.DT_FLOAT, None)
                B_reshape_3to2dim = op_reshape_output_name

                # called 2dim function
                A_2dim = A_reshape_3to2dim
                B_2dim = B_reshape_3to2dim
                A_shape_2dim = op_left_output_shape
                B_shape_2dim = op_right_output_shape
                output_2dim = op_name + '_output_2dim'
                output_shape_2dim = [A_shape_2dim[-2],B_shape_2dim[-1]]
                _Adims2xBdims2(op, op_name, A_2dim, B_2dim, A_shape_2dim, B_shape_2dim, output_2dim, output_shape_2dim)

                # create reshape to change 2 dim to 3 dim
                reshape_output_shape = output_shape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape3'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shape3'
                reshape_output_tensor_data = reshape_output_shape
                reshape_output_tensor_shape = [len(reshape_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                            mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([output_2dim])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_output3'
                op_reshape.output.extend([output])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
            elif B_dim_num == 3 and self.is_const_tensor(B) == True:
                if self.find_tensor_by_name(B).int32_data != []:
                    const_input_ori_data = self.find_tensor_by_name(B).int32_data
                else:
                    const_input_ori_data = self.find_tensor_by_name(B).float_data
                if len(const_input_ori_data) != 0:
                    data = np.array(const_input_ori_data)
                    data = data.reshape(B_shape)
                    data = data.reshape([B_shape[1],B_shape[2]])
                    data = list(data.flat)
                    new_data = copy.deepcopy(data)
                    const_tensor_name = op_name + '_constinput1'
                    const_tensor_shape = [B_shape[1],B_shape[2]]
                    const_tensor_data = new_data
                    self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                    mace_pb2.DT_FLOAT, const_tensor_data)
                    B = const_tensor_name

                # called 2dim function
                A_2dim = A_reshape_3to2dim
                B_2dim = B
                A_shape_2dim = op_left_output_shape
                B_shape_2dim = const_tensor_shape
                output_2dim = op_name + '_output_2dim'
                output_shape_2dim = [A_shape_2dim[-2],B_shape_2dim[-1]]
                _Adims2xBdims2(op, op_name, A_2dim, B_2dim, A_shape_2dim, B_shape_2dim, output_2dim, output_shape_2dim)

                # create reshape to change 2 dim to 3 dim
                reshape_output_shape = output_shape
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape3'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shape3'
                reshape_output_tensor_data = reshape_output_shape
                reshape_output_tensor_shape = [len(reshape_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                            mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([output_2dim])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_output3'
                op_reshape.output.extend([output])
                op_reshape.output_shape.add()
                op_reshape.output_shape[0].dims.extend(output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
            return op

        def _Adims4(op, op_name, A, B, A_shape, B_shape, output, output_shape):
            A_dim_num = len(A_shape)
            B_dim_num = len(B_shape)
            mace_check(A_shape[0] == A_shape[1] == 1,"the first and second dimension of the left operand are greater than 1 is not supported!")
            OutputNeedTrans = False
            if self.find_tensor_by_name(A).data_format == mace_pb2.DT_NCHW:
                OutputNeedTrans = True

            # A must be 4dim variable
            # 1\ transpose A if A transposed
            lhsChanged = False
            if self.is_const_tensor(A) == False:
                if self.find_tensor_by_name(A).data_format == mace_pb2.DT_NCHW:
                    op_transpose0 = self._SGSModel.op.add()
                    op_transpose0.name = op_name + '_transpose0'
                    op_transpose0.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose0.name + '_shape0'
                    shape_tensor_data = [0,3,1,2]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                    mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose0.input.extend([A])
                    op_transpose0.input.extend([shape_tensor_name])
                    output_op_transpose0 = op_transpose0.name + '_output0'
                    op_transpose0.output.extend([output_op_transpose0])
                    ### transpose input
                    tmp_dim = [0,3,1,2]
                    tmp_shape = [0,1,2,3]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = A_shape[tmp_dim[i]]
                    op_transpose0.output_shape.add()
                    op_transpose0_shape = copy.deepcopy(tmp_shape)
                    op_transpose0.output_shape[0].dims.extend(op_transpose0_shape)
                    new_A_shape = A_shape
                    lhsChanged = True
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose0)
                elif self.find_tensor_by_name(A).data_format == mace_pb2.DT_NHWC:
                    new_A_shape = A_shape
            else:
                mace_check(self.is_const_tensor(A) == False ,"only support variable tensor for 4dim MatMul lhs")

            # 2\ reshape A to 2dim
            left_op_output_shape = [1,1]
            left_op_output_shape[0] = new_A_shape[A_dim_num - 2]
            left_op_output_shape[1] = new_A_shape[A_dim_num - 1]
            op_reshape0 = self._SGSModel.op.add()
            op_reshape0.name = op_name + '_reshape0'
            op_reshape0.type = 'RESHAPE'
            reshape_output_tensor_name = op_reshape0.name + '_output_shape0'
            reshape_output_tensor_data = left_op_output_shape
            reshape_output_tensor_shape = [len(left_op_output_shape)]
            self.add_tensor(self._SGSModel, reshape_output_tensor_name,
                            reshape_output_tensor_shape, mace_pb2.DT_INT32, reshape_output_tensor_data)
            if lhsChanged == False:
                op_reshape0.input.extend([A])
            else:
                op_reshape0.input.extend([output_op_transpose0])
            op_reshape0.input.extend([reshape_output_tensor_name])
            op_reshape_output_name = op_reshape0.name + '_output0'
            op_reshape0.output.extend([op_reshape_output_name])
            op_reshape0.output_shape.add()
            op_reshape0.output_shape[0].dims.extend(left_op_output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_reshape0)
            A_reshape_4to2dim = op_reshape_output_name


            mace_check(B_dim_num == 2 and self.is_const_tensor(B) ,"only support 2 dims const tensor for 4dim MatMul rhs")
            if self.is_const_tensor(B) == True:
                if self.find_tensor_by_name(B).int32_data != []:
                    const_input_ori_data = self.find_tensor_by_name(B).int32_data
                else:
                    const_input_ori_data = self.find_tensor_by_name(B).float_data
                if len(const_input_ori_data) != 0:
                    data = np.array(const_input_ori_data)
                    data = data.reshape(B_shape)
                    data = data.transpose(1,0)
                    data = list(data.flat)
                    new_data = copy.deepcopy(data)
                    tmp_dim = [1,0]
                    tmp_shape = [0,1]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = B_shape[tmp_dim[i]]
                    rhsChanged = True
                    right_op_output_shape = B_shape
                    new_B = B

            # called 2dim function
            A_2dim = A_reshape_4to2dim
            B_2dim = new_B
            A_shape_2dim = left_op_output_shape
            B_shape_2dim = right_op_output_shape
            output_2dim = op_name + '_output_2dim'
            output_shape_2dim = [A_shape_2dim[-2],B_shape_2dim[-1]]
            _Adims2xBdims2(op, op_name, A_2dim, B_2dim, A_shape_2dim, B_shape_2dim, output_2dim, output_shape_2dim)

            # create reshape to change 2 dim to 4 dim
            if OutputNeedTrans == False:
                for outputName in op.output:
                    outputTensor = self.find_tensor_by_name(outputName)
                    outputTensor.data_format = mace_pb2.DT_NHWC
                    output = outputName
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape4'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shape4'
                reshape_output_tensor_data = output_shape[:]
                reshape_output_tensor_shape = [len(output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                            mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([output_2dim])
                op_reshape.input.extend([reshape_output_tensor_name])
                # op_reshape_output_name = op_reshape.name + '_output4'
                op_reshape.output.extend([output])
                op_reshape.output_shape.add()
                output_shape_4dim = output_shape
                op_reshape.output_shape[0].dims.extend(output_shape_4dim)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
            # trans output shape
            if OutputNeedTrans == True:
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape4'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shape4'
                reshape_output_tensor_data = output_shape[:]
                reshape_output_tensor_shape = [len(output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([output_2dim])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_output4'
                op_reshape.output.extend([op_reshape_output_name])
                op_reshape.output_shape.add()
                output_shape_4dim = [output_shape[0],output_shape[3],output_shape[1],output_shape[2]]
                op_reshape.output_shape[0].dims.extend(output_shape_4dim)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = op_name + '_transpose4'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape4'
                shape_tensor_data = [0,2,3,1]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([op_reshape_output_name])
                op_transpose.input.extend([shape_tensor_name])

                # transpose input
                tmp_dim = [0,2,3,1]
                tmp_shape = [0,1,2,3]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = output_shape_4dim[tmp_dim[i]]
                op_transpose.output.extend([output])
                op_transpose.output_shape.add()
                op_transpose_shape = copy.deepcopy(tmp_shape)
                op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)
















            return op

        # matmul function
        if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
            for op_ in self._SGSModel.op:
                if op_ == op:
                    op_name = op_.name
                    op_A = op_.input[0]              # left matrix name
                    op_B = op_.input[1]              # right  matrix name
                    op_output = op_.output[0]     # Matrix multiply results name from A * B
                    op_A_tensor = self.find_tensor_by_name(op_A)
                    op_B_tensor = self.find_tensor_by_name(op_B)
                    op_A_shape = op_A_tensor.dims
                    op_B_shape = op_B_tensor.dims
                    op_output_shape = self.find_tensor_by_name(op_output).dims # list
                    A_dim_num = len(op_A_tensor.dims)
                    mace_check(A_dim_num < 5 , "do not support more than 4 dims MatMul")

                    if A_dim_num == 4:
                        mace_check(self.is_const_tensor(op_A) == False , "do not support 4 dims MatMul whose lhs is const!")
                        _Adims4(op_, op_name, op_A, op_B, op_A_shape, op_B_shape, op_output, op_output_shape)
                    elif A_dim_num == 3:
                        mace_check(self.is_const_tensor(op_A) == False , "do not support 3 dims MatMul whose lhs is const!")
                        _Adims3(op_, op_name, op_A, op_B, op_A_shape, op_B_shape, op_output, op_output_shape)
                    elif A_dim_num == 2:
                        if self.is_const_tensor(op_A):
                            return self.split_LhsIsConstMatmul(op_)
                        _Adims2xBdims2(op_, op_name, op_A, op_B, op_A_shape, op_B_shape, op_output, op_output_shape)
                    self.remove_op_with_name(op_name)
        else:
            mace_check(getIPUVersion() == 'I6E' or getIPUVersion() == 'M6' , "The implementation version of Matmul does not match!")

    def split_LhsIsConstMatmul(self, op):
        # matmul A * B = C
        # Take the dimension of A as the benchmark
        #case 1: A:dims 2  B:dims 2
        #   switch 1:A:dims 2  const tensor
        #            B:dims 2  variable tensor
        def _split_2dim_LhsIsConstMatmul(op):
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
                        bias_data = np.zeros(C_shape[-1])
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
                            if self.find_tensor_by_name(bias_tensor_name).int32_data != []:
                                const_input_ori_data = self.find_tensor_by_name(bias_tensor_name).int32_data
                            else:
                                const_input_ori_data = self.find_tensor_by_name(bias_tensor_name).float_data
                            bias_data = np.array(const_input_ori_data)
                            bias_data = np.tile(bias_data, C_shape[-1])
                            bias_tensor_name = op_name + '_bias'
                            self.add_tensor(self._SGSModel,bias_tensor_name, bias_shape,
                                             mace_pb2.DT_FLOAT,bias_data)
                            op_.input[2] = bias_tensor_name
                            del bias_tensor
                    # 1. create new input[0]
                    if transA == True:
                        if self.find_tensor_by_name(A).int32_data != []:
                            const_input_ori_data = self.find_tensor_by_name(A).int32_data
                        else:
                            const_input_ori_data = self.find_tensor_by_name(A).float_data
                        if len(const_input_ori_data) != 0 and len(A_shape) == 2:
                            data = np.array(const_input_ori_data)
                            data = data.reshape(A_shape)
                            data = data.transpose(1,0)
                            data = list(data.flat)
                            new_data = copy.deepcopy(data)
                            tmp_dim = [1,0]
                            tmp_shape = [0,1]
                            for i in six.moves.range(len(tmp_dim)):
                                tmp_shape[i] = A_shape[tmp_dim[i]]
                            for i in six.moves.range(len(tmp_dim)):
                                A_tensor.dims[i] = tmp_shape[i]
                            const_tensor_name = op_name + '_constB'
                            const_tensor_shape = A_tensor.dims
                            const_tensor_data = data
                            self.add_tensor(self._SGSModel, const_tensor_name, const_tensor_shape,
                                            mace_pb2.DT_FLOAT, const_tensor_data)
                            new_A = const_tensor_name
                            del A_tensor
                    else:
                        new_A = A
                    # 2. create new input[1]
                    if transB == False:
                        if self.is_const_tensor(B) == False:
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
                            tmp_shape = [B_shape[1],B_shape[0]]
                            op_transpose.output_shape.add()
                            op_transpose.output_shape[0].dims.extend(tmp_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                            new_B = output_op_transpose
                    else:
                        new_B = B
                    # 3. create Fullyconnet
                    op_fc = self._SGSModel.op.add()
                    op_fc.name = op_name + '_fc'
                    op_fc.type = 'FULLY_CONNECTED'
                    op_fc.input.extend([new_B])
                    op_fc.input.extend([new_A])
                    op_fc.input.extend([bias_tensor_name])
                    output_op_fc = op_fc.name + '_output'
                    op_fc.output.extend([output_op_fc])
                    op_fc.output_shape.add()
                    output_shape = [C_shape[1], C_shape[0]]
                    op_fc.output_shape[0].dims.extend(output_shape)
                    self.add_tensor(self._SGSModel, output_op_fc, output_shape,
                                    mace_pb2.DT_FLOAT, None, data_fromat = mace_pb2.DT_NHWC)
                    self._maceOpArray = np.append(self._maceOpArray,op_fc)
                    # 4. create new output[0]
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = op_name + '_transpose_3dim_output'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = [1,0]
                    shape_tensor_shape = [2]
                    self.add_tensor(self._SGSModel, shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([output_op_fc])
                    op_transpose.input.extend([shape_tensor_name])
                    op_transpose.output.extend([C])
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(C_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)

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
                    else:
                        mace_check(False , "do not support more than 2 dims MatMul if left is const!")

    def split_Pad(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            pads_name = op_.input[1]
            pads = self.find_tensor_by_name(op_.input[1])
            input_tensor = self.find_tensor_by_name(op_.input[0])
            pads_list = []
            pads_back = len(pads.int32_data) // 2
            for i, j in zip(pads.int32_data[:pads_back], pads.int32_data[pads_back:]):
              pads_list.append([i, j])
            if pads_back == 4 and input_tensor.data_format == mace_pb2.DT_NCHW:
              tflite_c = pads_list[1]
              del pads_list[1]
              pads_list.append(tflite_c)
            pads.Clear()
            pads.name = pads_name
            pads.dims.extend([pads_back, 2])
            pads.data_format = mace_pb2.DT_NHWC
            pads.data_type = mace_pb2.DT_INT32
            for item in pads_list:
              pads.int32_data.extend(item)
            for arg in op_.arg:
              if arg.name == 'pads_mode':
                if arg.str == 'constant':
                  op_.type = "PAD"
                else:
                  mace_check(False, 'Pad only support `constant` mode!')

            if input_tensor.data_format == mace_pb2.DT_NHWC:
              for outputName in op_.output:
                outputTensor = self.find_tensor_by_name(outputName)
                outputTensor.data_format = mace_pb2.DT_NHWC
            self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Pooling(self, op):
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
                if len(inputShape) == 5:
                    return self.split_Pooling3D(op_)
                elif len(inputShape) == 3:
                    return self.split_Pooling1D(op_)
                if inputTensor.data_format == mace_pb2.DT_NHWC:
                    new_input_name = self.check_NCHW_And_Change(xi, op_name)
                    op_.input[0] = new_input_name
                arg = op_.arg

                # for AVGPool2d
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_padding_values_str:
                        paddingL,paddingR,paddingT,paddingB = arg[i].ints
                    elif name == 'count_include_pad':
                        count_include_pad_value = arg[i].i

                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_pooling_type_str:
                        pooling_type = arg[i].i
                        if pooling_type == 1:
                            op_.type = 'AVERAGE_POOL_2D'
                            # IPU100 AVGPool2d
                            if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
                                for i in six.moves.range(len(arg)):
                                    name = arg[i].name
                                    if name == MaceKeyword.mace_padding_values_str:
                                        paddingL,paddingR,paddingT,paddingB = arg[i].ints
                                    elif name == 'count_include_pad':
                                        count_include_pad_value = arg[i].i
                                if count_include_pad_value == 1 and paddingL != 0:
                                    # add pad op
                                    op_pad = self._SGSModel.op.add()
                                    op_pad.name = op_name + '_pad'
                                    op_pad.type = 'PAD'
                                    shape_tensor_name = op_pad.name + '_shape'
                                    shape_tensor_data = [0,0,paddingT,paddingB,paddingL,paddingR,0,0]
                                    shape_tensor_shape = [4,2]
                                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                                   mace_pb2.DT_INT32, shape_tensor_data)
                                    op_pad.input.extend([xi])
                                    op_pad.input.extend([shape_tensor_name])
                                    output_op_pad = op_pad.name + '_output'
                                    op_pad.output.extend([output_op_pad])
                                    op_pad.output_shape.add()
                                    [ni,ci,hi,wi] = self.get_shape_by_name(xi)
                                    output_shape_pad = [ni,ci,int(hi+2*paddingT),int(wi+2*paddingL)]
                                    op_pad.output_shape[0].dims.extend(output_shape_pad)
                                    self._maceOpArray = np.append(self._maceOpArray,op_pad)
                                    # modify AVGPool pad value
                                    for i in six.moves.range(len(arg)):
                                        name = arg[i].name
                                        if name == MaceKeyword.mace_padding_values_str:
                                            arg[i].ints[:] = [0,0,0,0]
                                    op_.input[0] = output_op_pad

                        elif pooling_type == 2:
                            op_.type = 'MAX_POOL_2D'
                        elif pooling_type == 3 or pooling_type == 4: #GlobalAveragePool,GlobalMaxPool=3,4
                            input_op_name = op_.input[0]
                            input_op_shape = self.get_shape_by_name(input_op_name)
                            kernel_h,kernel_w = input_op_shape[2:]
                            if pooling_type == 3:
                                op_.type = 'AVERAGE_POOL_2D'
                            elif pooling_type == 4:
                                op_.type = 'MAX_POOL_2D'
                            for i in six.moves.range(len(arg)):
                                name = arg[i].name
                                if name == MaceKeyword.mace_kernel_str:
                                    arg[i].ints[:] = []
                                    arg[i].ints.extend([kernel_h,kernel_w])
                # pooling output data format must be NCHW
                for outputName in op_.output:
                    outputTensor = self.find_tensor_by_name(outputName)
                    outputTensor.data_format = mace_pb2.DT_NCHW

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

                self._maceOpArray = np.append(self._maceOpArray,op_)

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
                op_transpose.name = op_name + '_transpose1'
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
                            op_pool3d.name = 'AvePool3D'+ op_name
                        elif pooling_type == 2:
                            op_pool3d.name = 'MaxPool3D'+ op_name
                        elif pooling_type == 3 or pooling_type == 4:
                            input_op_name = op_.input[0]
                            input_op_shape = self.get_shape_by_name(input_op_name)
                            kernel_d,kernel_h,kernel_w = input_op_shape[2:]
                            if pooling_type == 3:
                                op_pool3d.name = 'AvePool3D'+ op_name
                            elif pooling_type == 4:
                                op_pool3d.name = 'MaxPool3D'+ op_name
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

                # create transpose from NCHW back to NHWC
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
                op_transpose.name = op_name + '#transpose'
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
                op_reshape.output_shape[0].dims.extend([op_transpose.output_shape[0].dims[0], op_transpose.output_shape[0].dims[2],
                                                1, op_transpose.output_shape[0].dims[1]])  # nchw
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
                    elif name == 'count_include_pad':
                        count_include_pad_value = arg[i].i
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
                op_pool.output_shape[0].dims.extend([n,c,1,w])  # nchw
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
                op_transpose.name = op_name + '#transpose2'
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

    def split_Clip(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
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
                op_clip.name = 'clip'+ op_name
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
                op_clip.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_clip)


    def split_Reduce(self, op):
        xi = op.input[0]
        op_name = op.name
        input_shape = self.get_shape_by_name(op.input[0])
        inputTensor = self.find_tensor_by_name(xi)
        output_shape = op.output_shape[0].dims[:]
        arg = op.arg
        axis = -1
        reduce_type = "Unknow"
        axis_ori = 0
        axis_data = []
        ori_axis_data = []
        input = xi
        axis_exist = False

        #mace_check(len(input_shape) <= 4,'Reduce only supports maximumly 4 dims.')
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == MaceKeyword.mace_reduce_type_str:
                type = arg[i].i
                # MEAN = 0 MIN = 1 MAX = 2 PROD = 3 SUM = 4
                if type == 0:
                    reduce_type = "MEAN"
                elif type == 2:
                    reduce_type = 'REDUCE_MAX'
                elif type == 4:
                    reduce_type = 'SUM'
                else:
                    mace_check(False,'Reduce do not support.')
            if name == MaceKeyword.mace_axis_str: # "axis"
                for j in range(len(arg[i].ints)):
                    axis_exist = True
                    axis_ori = len(input_shape) + arg[i].ints[j] if arg[i].ints[j] < 0 else arg[i].ints[j]
                    if len(input_shape) == 4 and len(output_shape) == 4 and inputTensor.data_format == mace_pb2.DT_NCHW:#nchw -> nhwc
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
                    ori_axis_data.append(axis_ori)
                    ori_axis_data.sort()
                    axis_data.append(axis_new)
                    axis_data.sort()
            if name == MaceKeyword.mace_keepdims_str: # "keepdims"
                keepdim = arg[i].i

        # if axis = None
        if axis_exist == False:
            if len(input_shape) == 4:
                if len(output_shape) == 4 and inputTensor.data_format == mace_pb2.DT_NCHW:
                    axis_data = [0,1,2,3]
                    ori_axis_data = [0,1,2,3]
                else:
                    axis_data = [0,1,2,3]
                    ori_axis_data = [0,1,2,3]
            else:
                axis_data = [x for x in range(len(input_shape))]
                ori_axis_data = [x for x in range(len(input_shape))]

        if len(input_shape) == 4 and len(output_shape) != 4:
            # when reduce out shape won't change after NCHW-NHWC transpose
            # no need to create transpose to restore original dims shape
            need_add_transpose = False
            if inputTensor.data_format == mace_pb2.DT_NCHW:
                shape_order = [0,2,3,1]
                for i in axis_data:
                    shape_order.remove(i)
                if len(shape_order) > 1:
                    for i in range(len(shape_order)-1):
                        if shape_order[i] > shape_order[i+1]:
                            need_add_transpose = True
                else:
                    need_add_transpose = True
                if not need_add_transpose:
                    for i in range(len(axis_data)):
                        axis_data[i] = self.get_axes_by_shape(axis_data[i],len(input_shape))
                    axis_data.sort()

            # need create transpose to restore original dims shape
            if len(input_shape) == 4 and inputTensor.data_format == mace_pb2.DT_NCHW and need_add_transpose:
                # creat transpose for NHWC to NCHW
                output_shape = copy.deepcopy(input_shape)
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = op_name + '_transpose'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0,3,1,2]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([input])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                tmp_dim = [0,3,1,2]
                tmp_shape = [0,1,2,3]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = output_shape[tmp_dim[i]]
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(tmp_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                input = output_op_transpose

            # creat axis tensor
            axis_tensor_name = op_name + '_axis'
            axis_tensor_data = axis_data
            axis_tensor_shape = [len(axis_data)]
            self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                            mace_pb2.DT_INT32, axis_tensor_data)

            # creat ori reduce operator
            if reduce_type == "REDUCE_MAX" or reduce_type == "SUM":
                op_reduceOp = self._SGSModel.op.add()
                op_reduceOp.name = op_name + '_reduceSum'
                op_reduceOp.type = reduce_type
                keepdims_arg = op_reduceOp.arg.add()
                keepdims_arg.name = 'keepdims'
                keepdims_arg.i = keepdim
                op_reduceOp.input.extend([input])
                op_reduceOp.input.extend([axis_tensor_name])
                output_op_reduceOp = op_reduceOp.name + '_output'
                op_reduceOp.output[:] =  op.output[:]
                op_reduceOp.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reduceOp)
                self.remove_op_with_name(op_name)
            elif reduce_type == "MEAN":
                # SGS MEAN only support H W
                if axis_data == [1,2]:
                    for op_ in self._SGSModel.op:
                        if op_ == op:
                            op_.type = "MEAN"
                            op_.input.pop()
                            op_.input.extend([input])
                            op_.input.extend([axis_tensor_name])
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                else:
                    #reducemean = reduceSum % dimNum
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
                    op_reduceSum.output_shape.extend(op.output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)
                    #creat div
                    for i in range(len(ori_axis_data)):
                        axis_new = ori_axis_data[i]
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
                    op_div.output[:] =  op.output[:]
                    op_div.output_shape.extend(op.output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_div)
                    self.remove_op_with_name(op_name)
        # keepDims = True; no need to create transpose to restore original dims shape
        else:
            for op_ in self._SGSModel.op:
                if op_ == op:
                    op_.type = reduce_type
                    # creat axis tensor
                    axis_tensor_name = op_name + '_axis'
                    axis_tensor_data = axis_data
                    axis_tensor_shape = [len(axis_data)]
                    self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                    mace_pb2.DT_INT32, axis_tensor_data)
                    #reducemean = reduceSum % dimNum
                    if reduce_type == "MEAN":
                        # SGS MEAN only support H W
                        if axis_data == [1,2]:
                            op_.type = "MEAN"
                            op_.input.extend([axis_tensor_name])
                            self._maceOpArray = np.append(self._maceOpArray,op_)
                            return
                        else:
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
                            if len(output_shape) == 4 and inputTensor.data_format == mace_pb2.DT_NHWC:
                                tmp_dim = [0,3,1,2]
                                tmp_shape = [0,1,2,3]
                                for i in six.moves.range(len(tmp_dim)):
                                    tmp_shape[i] = output_shape[tmp_dim[i]]
                            else:
                                tmp_shape = output_shape
                            op_reduceSum.output_shape.add()
                            op_reduceSum.output_shape[0].dims.extend(tmp_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)
                            #creat div
                            for i in range(len(ori_axis_data)):
                                axis_new = ori_axis_data[i]
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
                            op_div.output[:] =  op.output[:]
                            op_div.output_shape.extend(op.output_shape)
                            self._maceOpArray = np.append(self._maceOpArray,op_div)
                            self.remove_op_with_name(op_name)
                            return

                    op_.input.extend([axis_tensor_name])
                    self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Reshape(self, op):
        xi = op.input[0]
        op_name = op.name
        inputTensor = self.find_tensor_by_name(xi)
        input_shape = self.get_shape_by_name(xi)
        outputTensor = self.find_tensor_by_name(op.output[0])
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

        if inputTensor.data_format == mace_pb2.DT_NHWC:
            for op_ in self._SGSModel.op:
                if op_ == op:
                    #output_shape = op_.output_shape[0].dims[:]
                    output_tensor_name = op_.name + '_output_shape'
                    output_tensor_data = output_shape
                    output_tensor_shape = [len(output_shape)]
                    self.add_tensor(self._SGSModel, output_tensor_name, output_tensor_shape,
                                    mace_pb2.DT_INT32, output_tensor_data,mace_pb2.DT_NHWC)
                    op_.input.pop()
                    op_.input.extend([output_tensor_name])
                    op_.type = "RESHAPE"
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                    for outputName in op_.output:
                        outputTensor = self.find_tensor_by_name(outputName)
                        outputTensor.data_format = mace_pb2.DT_NHWC
        else:
            reshape_input_name = xi

            if len(input_shape) == 4 and (input_shape_total not in input_shape):
                # creat transpose for NHWC to NCHW in tflite model
                op_transpose = self._SGSModel.op.add()
                op_transpose.name = op_name + '_transpose'
                op_transpose.type = 'TRANSPOSE'
                shape_tensor_name = op_transpose.name + '_shape'
                shape_tensor_data = [0,3,1,2]
                shape_tensor_shape = [4]
                self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                mace_pb2.DT_INT32, shape_tensor_data)
                op_transpose.input.extend([xi])
                op_transpose.input.extend([shape_tensor_name])
                output_op_transpose = op_transpose.name + '_output'
                op_transpose.output.extend([output_op_transpose])
                tmp_dim = [0,3,1,2]# NCHW --[0312]--> NWCH  --[0231]--> NCHW
                tmp_shape = [0,1,2,3]
                for i in six.moves.range(len(tmp_dim)):
                    tmp_shape[i] = input_shape[tmp_dim[i]]
                op_transpose.output_shape.add()
                op_transpose.output_shape[0].dims.extend(tmp_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                self.name_shape_map[output_op_transpose] = tmp_shape
                reshape_input_name = output_op_transpose

            if len(output_shape) == 4:
                if output_shape_total not in output_shape:
                    # creat ori reshape layer
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape'
                    op_reshape.type = 'RESHAPE'
                    reshape_shape_tensor_name = op_reshape.name + '_output_shape'
                    reshape_shape_tensor_data = output_shape
                    reshape_shape_tensor_shape = [len(output_shape)]
                    self.add_tensor(self._SGSModel, reshape_shape_tensor_name, reshape_shape_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_shape_tensor_data)
                    op_reshape.input.extend([reshape_input_name])
                    op_reshape.input.extend([reshape_shape_tensor_name])
                    op_reshape_output_name = op_reshape.name + '_output'
                    op_reshape.output.extend([op_reshape_output_name])
                    op_reshape.output_shape.add()
                    [rn,rc,rh,rw] = output_shape
                    op_reshape.output_shape[0].dims.extend([rn,rw,rc,rh])
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)

                    # creat transpose for NCHW to NHWC in tflite model
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = op_name + '_transpose#1'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = [0,2,3,1]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                        mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([op_reshape_output_name])
                    op_transpose.input.extend([shape_tensor_name])
                    output_op_transpose = op_transpose.name + '_output'
                    op_transpose.output[:] =  op.output[:]
                    op_transpose.output_shape.extend(op.output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                else:
                    # creat reshape layer
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape'
                    op_reshape.type = 'RESHAPE'
                    reshape_shape_tensor_name = op_reshape.name + '_output_shape'
                    [rn,rc,rh,rw] = output_shape
                    reshape_shape_tensor_data = [rn,rh,rw,rc]
                    reshape_shape_tensor_shape = [len(output_shape)]
                    self.add_tensor(self._SGSModel, reshape_shape_tensor_name, reshape_shape_tensor_shape,
                                    mace_pb2.DT_INT32, reshape_shape_tensor_data)
                    op_reshape.input.extend([reshape_input_name])
                    op_reshape.input.extend([reshape_shape_tensor_name])
                    op_reshape.output[:] =  op.output[:]
                    op_reshape.output_shape.extend(op.output_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_reshape)
            else:
                # creat ori reshape layer
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape'
                op_reshape.type = 'RESHAPE'
                reshape_shape_tensor_name = op_reshape.name + '_output_shape'
                reshape_shape_tensor_data = output_shape
                reshape_shape_tensor_shape = [len(output_shape)]
                self.add_tensor(self._SGSModel, reshape_shape_tensor_name, reshape_shape_tensor_shape,
                                mace_pb2.DT_INT32, reshape_shape_tensor_data)
                op_reshape.input.extend([reshape_input_name])
                op_reshape.input.extend([reshape_shape_tensor_name])
                op_reshape.output[:] =  op.output[:]
                op_reshape.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_reshape)

            self.remove_op_with_name(op_name)

    def split_Slice(self, op):
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

            # switch dims for begin/end/stride
            if input_tensor.data_format == mace_pb2.DT_NCHW and len(input_shape) == 4:
              value_c = begin_tensor[1]
              del begin_tensor[1]
              begin_tensor.append(value_c)

              value_c = end_tensor[1]
              del end_tensor[1]
              end_tensor.append(value_c)

              value_c = strides_tensor[1]
              del strides_tensor[1]
              strides_tensor.append(value_c)

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

            if input_tensor.data_format == mace_pb2.DT_NHWC:
              for outputName in op_.output:
                outputTensor = self.find_tensor_by_name(outputName)
                outputTensor.data_format = mace_pb2.DT_NHWC
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Split(self, op):
        xi = op.input[0]
        slice_point_enable = False
        inputTensor = self.find_tensor_by_name(xi)
        use_slice = False
        output_detached = []
        for i in six.moves.range(len(op.output)):
            if self.is_tensor_detached(op.output[i]):
                use_slice = True
                output_detached.extend([1])
            else:
                output_detached.extend([0])

        if use_slice == False:
          if inputTensor.data_format == mace_pb2.DT_NHWC:
            for op_ in self._SGSModel.op:
                 if op_ == op:
                   arg = op_.arg
                   output_shape = op_.output_shape[:]
                   op_input = op_.input[0]
                   for i in six.moves.range(len(arg)):
                     name = arg[i].name
                     if name == MaceKeyword.mace_axis_str:
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
                   if slice_point_enable: # creat split_v
                       op_.input.extend([slice_point_name])
                       op_.input.extend([axis_tensor_name])
                       op_.type = "SPLIT_V"
                   else:
                       op_.input[0] = axis_tensor_name
                       op_.input.extend([op_input])
                       op_.type = "SPLIT"
                   self._maceOpArray = np.append(self._maceOpArray, op_)
                   for outputName in op_.output:
                     outputTensor = self.find_tensor_by_name(outputName)
                     outputTensor.data_format = mace_pb2.DT_NHWC
          else:
            for op_ in self._SGSModel.op:
                if op_ == op:
                  arg = op_.arg
                  output_shape = op_.output_shape[:]
                  op_input = op_.input[0]
                  inputShape = self.get_shape_by_name (op_input)
                  for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_axis_str:
                      arg[i].i = self.get_axes_by_shape (arg[i].i, len(inputShape))
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
        else:
            for op_ in self._SGSModel.op:
                if op_ == op:
                    arg = op_.arg
                    output_shape = op_.output_shape[:]
                    op_input = op_.input[0]
                    inputShape = self.get_shape_by_name (op_input)
                    for i in six.moves.range(len(arg)):
                        name = arg[i].name
                        if name == MaceKeyword.mace_axis_str:
                            axis = arg[i].i
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
                            if len(inputShape) == 4 and inputTensor.data_format == mace_pb2.DT_NCHW:
                                tmp = begin_tensor_data[1]
                                begin_tensor_data[1] = begin_tensor_data[2]
                                begin_tensor_data[2] = begin_tensor_data[3]
                                begin_tensor_data[3] = tmp
                            self.add_tensor(self._SGSModel, begin_tensor_name, begin_tensor_shape,
                                            mace_pb2.DT_INT32, begin_tensor_data)
                            size_tensor_name = op_slice.name + '_size'
                            size_tensor_shape = [len(inputShape)]
                            if len(inputShape) == 4 and inputTensor.data_format == mace_pb2.DT_NCHW:
                                tmp = size_tensor_data[1]
                                size_tensor_data[1] = size_tensor_data[2]
                                size_tensor_data[2] = size_tensor_data[3]
                                size_tensor_data[3] = tmp
                            self.add_tensor(self._SGSModel, size_tensor_name, size_tensor_shape,
                                            mace_pb2.DT_INT32, size_tensor_data)
                            op_slice.input.extend([xi])
                            op_slice.input.extend([begin_tensor_name])
                            op_slice.input.extend([size_tensor_name])
                            op_slice.output.extend([op_.output[i]])
                            op_slice.output_shape.add()
                            op_slice.output_shape[0].dims.extend(size_tensor_data)
                            self._maceOpArray = np.append(self._maceOpArray,op_slice)

                    if inputTensor.data_format == mace_pb2.DT_NHWC:
                        for outputName in op_.output:
                            outputTensor = self.find_tensor_by_name(outputName)
                            outputTensor.data_format = mace_pb2.DT_NHWC
                    self.remove_op_with_name(op_.name)

    def split_Softmax(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            xi = op.input[0]
            op_name = op.name
            slice_point_enable = False
            inputTensor = self.find_tensor_by_name(xi)
            inputShape = self.get_shape_by_name(xi)
            output_shape = op.output_shape[0].dims[:]
            arg = op.arg
            shape_tensor_data1 = [x for x in range(0, len(inputShape))]
            shape_tensor_data2 = [x for x in range(0, len(inputShape))]

            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == 'axis':
                arg[i].i = len(output_shape) + arg[i].i if arg[i].i < 0 else arg[i].i
                axis= arg[i].i
                mace_check(axis < len(output_shape) and axis >= 0,
                          " axis must be within length of input dims\n")

            # SGS Softmax only do softmax on innermost dim, so need to transpose according to axis param

            # In this case, as we will do auto transpose from NCHW to NHWC,
            # axis 1 will be transposed directly to innermost dim. No need for special treatment
            if inputTensor.data_format == mace_pb2.DT_NCHW and len(inputShape) == 4 and axis == 1:
              op_.type = "SOFTMAX"
              self._maceOpArray = np.append(self._maceOpArray,op_)

            # In this case, as we won't do auto transpose from NCHW to NHWC,
            # last axis will be keeped at innermost dim. No need for special treatment
            elif (inputTensor.data_format == mace_pb2.DT_NHWC or len(inputShape) != 4) and axis == len(inputShape) -1:
              op_.type = "SOFTMAX"
              if inputTensor.data_format == mace_pb2.DT_NHWC:
                for outputName in op_.output:
                  outputTensor = self.find_tensor_by_name(outputName)
                  outputTensor.data_format = mace_pb2.DT_NHWC
              self._maceOpArray = np.append(self._maceOpArray,op_)
            # For other cases, transpose dim pointed by
            else:
              if inputTensor.data_format == mace_pb2.DT_NCHW and len(inputShape) == 4:
                shape_tensor_data1 = [0,3,1,2]
                temp = shape_tensor_data1[axis]
                shape_tensor_data1[axis] = shape_tensor_data1[-1]
                shape_tensor_data1[-1] = temp

                tmp_data = [0,1,2,3]
                temp = tmp_data[axis]
                tmp_data[axis] = tmp_data[-1]
                tmp_data[-1] = temp
                tmp_dim = [0,2,3,1]
                tmp_data2= [0,1,2,3]
                for i in six.moves.range(len(tmp_dim)):
                  tmp_data2[i] = tmp_data[tmp_dim[i]]
                shape_tensor_data2=tmp_data2
              else:
                temp = shape_tensor_data1[axis]
                shape_tensor_data1[axis] = shape_tensor_data1[-1]
                shape_tensor_data1[-1] = temp
                shape_tensor_data2 = shape_tensor_data1

              # creat transpose for switch axis to innermost dim
              op_transpose = self._SGSModel.op.add()
              op_transpose.name = op_name + '_transpose'
              op_transpose.type = 'TRANSPOSE'
              shape_tensor_name = op_transpose.name + '_shape1'
              shape_tensor_shape = [len(inputShape)]
              self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                  mace_pb2.DT_INT32, shape_tensor_data1)
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
              if len(inputShape) == 4:
                tmp_dim = [0,3,1,2]
                tmp_shape2 = [0,1,2,3]
                for i in six.moves.range(len(tmp_dim)):
                  tmp_shape2[i] = tmp_shape[tmp_dim[i]]
                tmp_shape = tmp_shape2

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
                  mace_pb2.DT_INT32, shape_tensor_data2)
              op_transpose2.input.extend([output_op_softmax])
              op_transpose2.input.extend([shape_tensor_name2])
              op_transpose2.output.extend(op.output)
              op_transpose2.output_shape.add()
              op_transpose2.output_shape[0].dims.extend(output_shape)

              if inputTensor.data_format == mace_pb2.DT_NHWC:
                for outputName in op_.output:
                  outputTensor = self.find_tensor_by_name(outputName)
                  outputTensor.data_format = mace_pb2.DT_NHWC
              self._maceOpArray = np.append(self._maceOpArray,op_transpose2)
              self.remove_op_with_name(op_name)

    def split_Softplus(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                op_.type = 'CUSTOM'
                op_.name = 'Softplus'+ op_name
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Transpose(self, op):
        op_name = op.name
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        input_shape = self.get_shape_by_name(xi)
        intputShapeSize = len(input_shape)
        output_shape = op.output_shape[0].dims
        if inputTensor.data_format == mace_pb2.DT_NCHW and intputShapeSize == 4:
          [nn,nc,nh,nw] = output_shape
          arg = op.arg
          for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == MaceKeyword.mace_dims_str:
                [n,c,h,w] = arg[i].ints
          # creat transpose for NHWC to NCHW
          op_transpose = self._SGSModel.op.add()
          op_transpose.name = op_name + '_transpose'
          op_transpose.type = 'TRANSPOSE'
          shape_tensor_name = op_transpose.name + '_shape'
          shape_tensor_data = [0,3,1,2]
          shape_tensor_shape = [4]
          self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
              mace_pb2.DT_INT32, shape_tensor_data)
          op_transpose.input.extend([xi])
          op_transpose.input.extend([shape_tensor_name])
          output_op_transpose = op_transpose.name + '_output'
          op_transpose.output.extend([output_op_transpose])
          tmp_dim = [0,3,1,2]
          tmp_shape = [0,1,2,3]
          for i in six.moves.range(len(tmp_dim)):
            tmp_shape[i] = input_shape[tmp_dim[i]]
          op_transpose.output_shape.add()
          op_transpose.output_shape[0].dims.extend(tmp_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_transpose)

          # creat transpose follow ori layer parameter
          op_transpose = self._SGSModel.op.add()
          op_transpose.name = op_name + '_transpose#1'
          op_transpose.type = 'TRANSPOSE'
          shape_tensor_name = op_transpose.name + '_shape'
          shape_tensor_data = [n,c,h,w]
          shape_tensor_shape = [4]
          self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
              mace_pb2.DT_INT32, shape_tensor_data)
          op_transpose.input.extend([output_op_transpose])
          op_transpose.input.extend([shape_tensor_name])
          op_transpose.output.extend(op.output)
          op_transpose.output_shape.add()
          op_transpose.output_shape[0].dims.extend([nn,nw,nc,nh])
          self._maceOpArray = np.append(self._maceOpArray,op_transpose)
          self.remove_op_with_name(op_name)

          # modify outputTensor data_format
          for outputName in op.output:
             outputTensor = self.find_tensor_by_name(outputName)
             outputTensor.data_format = mace_pb2.DT_NHWC

        else:
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
              for outputName in op.output:
                outputTensor = self.find_tensor_by_name(outputName)
                outputTensor.data_format = mace_pb2.DT_NHWC
              self._maceOpArray = np.append(self._maceOpArray,op_)


    def split_Tile(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "TILE"
            self._maceOpArray = np.append(self._maceOpArray,op_)


    def split_Normalize(self, op):
        across_spatial,channel_shared = 1,1
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
          #add padding_type
          paddingType_arg = op_conv.arg.add()
          paddingType_arg.name = MaceKeyword.mace_padding_str
          #set padding_type
          paddingType_arg.i = self.pooling_paddingType['CAFFE']
          strides_arg = op_conv.arg.add()
          strides_arg.name = 'strides'
          strides_arg.ints.extend([1,1])
          padding_arg = op_conv.arg.add()
          padding_arg.name = 'padding_values'
          padding_arg.ints.extend([0,0,0,0])
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
          #add padding_type
          paddingType_arg = op_conv.arg.add()
          paddingType_arg.name = MaceKeyword.mace_padding_str
          #set padding_type
          paddingType_arg.i = self.pooling_paddingType['CAFFE']
          strides_arg = op_conv.arg.add()
          strides_arg.name = 'strides'
          strides_arg.ints.extend([1,1])
          padding_arg = op_conv.arg.add()
          padding_arg.name = 'padding_values'
          padding_arg.ints.extend([0,0,0,0])
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

    def split_Not(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "LOGICAL_NOT"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_And(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "LOGICAL_AND"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Reorg(self, op):
        #only support strid is 2
        [n,c,h,w] = op.output_shape[0].dims[:]
        c = int(c/4)
        op_name = op.name
        xi = op.input[0]

        #creat transpose
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = op_name + '_transpose'
        op_transpose.type = 'TRANSPOSE'
        shape_tensor_name = op_transpose.name + '_shape'
        shape_tensor_data = [0,3,1,2]
        shape_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
            mace_pb2.DT_INT32, shape_tensor_data)
        op_transpose.input.extend([xi])
        op_transpose.input.extend([shape_tensor_name])
        output_op_transpose = op_transpose.name + '_output'
        op_transpose.output.extend([output_op_transpose])
        op_transpose.output_shape.add()
        op_transpose.output_shape[0].dims.extend([n,int(2*w),c,int(2*h)])
        self._maceOpArray = np.append(self._maceOpArray,op_transpose)

        #creat reshape
        op_reshape = self._SGSModel.op.add()
        op_reshape.name = op_name + '_reshape'
        op_reshape.type = 'RESHAPE'
        op_reshape.input.extend([output_op_transpose])
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

        output_op_reshape = op_reshape.name + '_output'
        op_reshape.output.extend([output_op_reshape])
        op_reshape.output_shape.add()
        op_reshape.output_shape[0].dims.extend([n,w,int(4*c),h])
        self._maceOpArray = np.append(self._maceOpArray,op_reshape)

        #creat transpose
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = op_name + '_transpose#2'
        op_transpose.type = 'TRANSPOSE'
        shape_tensor_name = op_transpose.name + '_shape'
        shape_tensor_data = [0,2,3,1]
        shape_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
            mace_pb2.DT_INT32, shape_tensor_data)
        op_transpose.input.extend([output_op_reshape])
        op_transpose.input.extend([shape_tensor_name])
        op_transpose.output[:] =  op.output[:]
        op_transpose.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_transpose)

        #remove op
        self.remove_op_with_name(op_name)
    def split_Upsample(self, op):
        [n,c,h,w] = op.output_shape[0].dims[:]
        [ni,ci,hi,wi] = self.get_shape_by_name(op.input[0])
        op_name = op.name
        xi = op.input[0]#NHWC in interpreter
        scale = 1
        for op_ in self._SGSModel.op:
          if op_ == op:
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
        op_tile.output_shape[0].dims.extend([ni,ci*scale,hi,wi*scale])
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
        op_reshape.output_shape.extend(op.output_shape)
        op_reshape.output[:] =  op.output[:]
        self._maceOpArray = np.append(self._maceOpArray,op_reshape)

        #remove op
        self.remove_op_with_name(op_name)


    def split_Threshold(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "GREATER_EQUAL"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Reduce_Mean(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "MEAN"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Cos(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "COS"
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Sin(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "SIN"
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Less(self, op):
        # Transposed
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "LESS"
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_ResizeBilinear(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "RESIZE_BILINEAR"
              #del op_.input[1]
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_ResizeNearestNeighbor(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "RESIZE_NEAREST_NEIGHBOR"
              #del op_.input[1]
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_ReduceL2(self,op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op_.name
                xi = op_.input[0]
                inputTensor = self.find_tensor_by_name(xi)
                output = op_.output[0]
                outputTensor = self.find_tensor_by_name(output)
                [n,c,h,w] = input_shape = self.get_shape_by_name(xi)
                output_shape = op.output_shape[0].dims[:]
                arg = op_.arg
                power = 2
                axis_ori = 0
                axis_data = []
                ori_axis_data = []
                need_transpose = False
                keepdim = 1
                # creat transpose change back to NCHW
                if len(input_shape) == 4 and inputTensor.data_format == mace_pb2.DT_NCHW:
                    need_transpose = True
                    output_shape = copy.deepcopy(input_shape)
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = op_name + '_transpose'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shape'
                    shape_tensor_data = [0,3,1,2]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                                    mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([input])
                    op_transpose.input.extend([shape_tensor_name])
                    output_op_transpose = op_transpose.name + '_output'
                    op_transpose.output.extend([output_op_transpose])
                    tmp_dim = [0,3,1,2]
                    tmp_shape = [0,1,2,3]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = output_shape[tmp_dim[i]]
                    op_transpose.output_shape.add()
                    op_transpose.output_shape[0].dims.extend(tmp_shape)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    input = output_op_transpose

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
                                    mace_pb2.DT_FLOAT, None, data_fromat = mace_pb2.DT_NHWC)
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
                                mace_pb2.DT_FLOAT, None, data_fromat = mace_pb2.DT_NHWC)
                self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)

                # creat sqrt
                op_sqrt = self._SGSModel.op.add()
                op_sqrt.name = op_name + '_sqrt'
                op_sqrt.type = 'SQRT'
                op_sqrt.input.extend([output_op_reduceSum])
                #op_sqrt.input.extend([const_tensor_name])
                if len(input_shape) == 4 and inputTensor.data_format == mace_pb2.DT_NCHW:
                    output_op_sqrt = op_sqrt.name + '_output'
                    op_sqrt.output.extend([output_op_sqrt])
                    op_sqrt.output_shape.add()
                    op_sqrt.output_shape[0].dims.extend(output_shape)
                else:
                    op_sqrt.output[:] =  op.output[:]
                    op_sqrt.output_shape.extend(op.output_shape)
                self._maceOpArray = np.append(self._maceOpArray,op_sqrt)

                if len(input_shape) == 4 and inputTensor.data_format == mace_pb2.DT_NCHW:
                    op_transpose = self._SGSModel.op.add()
                    op_transpose.name = op_name + '_transposeOut'
                    op_transpose.type = 'TRANSPOSE'
                    shape_tensor_name = op_transpose.name + '_shapeOut'
                    shape_tensor_data = [0,2,3,1]
                    shape_tensor_shape = [4]
                    self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape, mace_pb2.DT_INT32, shape_tensor_data)
                    op_transpose.input.extend([output_op_sqrt])
                    op_transpose.input.extend([shape_tensor_name])
                    # transpose input
                    tmp_dim = [0,2,3,1]
                    tmp_shape = [0,1,2,3]
                    for i in six.moves.range(len(tmp_dim)):
                        tmp_shape[i] = output_shape[tmp_dim[i]]
                    op_transpose.output[:] =  op.output[:]
                    op_transpose.output_shape.add()
                    op_transpose_shape = copy.deepcopy(tmp_shape)
                    op_transpose.output_shape[0].dims.extend(op_transpose_shape)
                    # self.add_tensor(self._SGSModel, C, op_transpose_shape, mace_pb2.DT_FLOAT,
                    #                 None, data_fromat = mace_pb2.DT_NHWC)
                    self._maceOpArray = np.append(self._maceOpArray,op_transpose)
                    self.remove_op_with_name(op_name)


    def split_SpaceToDepth(self, op):
        xi = op.input[0]
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'block_size':
            block_size = arg[i].i
            mace_check((block_size == 2),
                      "only support block_size num is 2 yet")
        [n,c,h,w] = self.get_shape_by_name(xi)
        c1 = int(c*block_size*block_size)
        w1 = w//block_size
        h1 = h//block_size
        op_name = op.name
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
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = op_name + '_transpose#1'
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
        '''
        #creat transpose
        op_transpose = self._SGSModel.op.add()
        op_transpose.name = op_name + '_transpose#2'
        op_transpose.type = 'TRANSPOSE'
        shape_tensor_name = op_transpose.name + '_shape'
        shape_tensor_data = [0,2,3,1]
        shape_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
            mace_pb2.DT_INT32, shape_tensor_data)
        op_transpose.input.extend([output_op_reshape])
        op_transpose.input.extend([shape_tensor_name])
        op_transpose.output[:] =  op.output[:]
        op_transpose.output_shape.add()
        op_transpose.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_transpose)
        '''
        self.remove_op_with_name(op_name)

    def split_SGS_SSD_Postprocess(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "CUSTOM"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_SGS_YoloV2_Postprocess(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "CUSTOM"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_SGS_YoloV3_Postprocess(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "CUSTOM"
                self._maceOpArray = np.append(self._maceOpArray,op_)

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

    def split_QIRQuantize(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_name = op.name
                op_.type = "CUSTOM"
                op_.name = 'QIRQuantize' + op_name
                self._maceOpArray = np.append(self._maceOpArray,op_)
