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
from mace.python.tools.converter_tool.transformer import Transformer
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.onnx import SGSModel_transform_onnx
from mace.proto import mace_pb2
from third_party.python import flatbuffers
from third_party import tflite

INVALID_FILTER = False


class ConvertSGSModel(object):
    #def __init__(self, net, input_nodes, output_nodes):
    def __init__(self, net, inputName, inputShape, outputName, outputDir, input_pack_model):
        self._net = net
        self._netInputName = inputName
        self._netInputShape = inputShape
        self._netOutputName = outputName
        self._outputDir = outputDir
        self._tensorArray= np.array([])
        self._opNodeArray = np.array([])
        self._maceOpArray = np.array([])
        self._mapBufferToTensor = {}
        self._mapTensorToSubgraph = {}
        self._operatorCodeArray = collections.OrderedDict()
        self._builder = flatbuffers.Builder(1024)
        self._input_pack_model = input_pack_model
        self._builtin_op = {
            'ABS':self.builtin_ABS,
            'ADD':self.builtin_ADD,
            'ARG_MAX':self.builtin_ARG_MAX,
            'MAX_POOL_2D':self.builtin_MAX_POOL_2D,
            'AVERAGE_POOL_2D':self.builtin_AVERAGE_POOL_2D,
            'CONCATENATION':self.builtin_CONCATENATION,
            'CONV_2D':self.builtin_CONV_2D,
            'DEPTHWISE_CONV_2D':self.builtin_DEPTHWISE_CONV_2D,
            'DEQUANTIZE':self.builtin_DEQUANTIZE,
            'EMBEDDING_LOOKUP':self.builtin_EMBEDDING_LOOKUP,
            'FLOOR':self.builtin_FLOOR,
            'Activation':self.builtin_ACTIVATION,
            'FULLY_CONNECTED':self.builtin_FULLY_CONNECTED,
            'HASHTABLE_LOOKUP':self.builtin_HASHTABLE_LOOKUP,
            'L2_NORMALIZATION':self.builtin_L2_NORMALIZATION,
            'L2_POOL_2D':self.builtin_L2_POOL_2D,
            'LOGISTIC':self.builtin_LOGISTIC,
            'LSH_PROJECTION':self.builtin_LSH_PROJECTION,
            'LSTM':self.builtin_LSTM,
            'MUL':self.builtin_MUL,
            'RELU':self.builtin_RELU,
            'LEAKY_RELU':self.builtin_LEAKY_RELU,
            'RELU6':self.builtin_RELU6,
            'RESHAPE':self.builtin_RESHAPE,
            'RESIZE_BILINEAR':self.builtin_RESIZE_BILINEAR,
            'RNN':self.builtin_RNN,
            'SOFTMAX':self.builtin_SOFTMAX,
            'SPACE_TO_DEPTH':self.builtin_SPACE_TO_DEPTH,
            'SVDF':self.builtin_SVDF,
            # 'LstmNonlinear',
            'TANH':self.builtin_TANH,
            'CONCAT_EMBEDDINGS':self.builtin_CONCAT_EMBEDDINGS,
            'CALL':self.builtin_CALL,
            'EMBEDDING_LOOKUP_SPARSE':self.builtin_EMBEDDING_LOOKUP_SPARSE,
            'PAD':self.builtin_PAD,
            'UNIDIRECTIONAL_SEQUENCE_RNN':self.builtin_UNIDIRECTIONAL_SEQUENCE_RNN,
            'GATHER':self.builtin_GATHER,
            'BATCH_TO_SPACE_ND':self.builtin_BATCH_TO_SPACE_ND,
            'SPACE_TO_BATCH_ND':self.builtin_SPACE_TO_BATCH_ND,
            'TRANSPOSE':self.builtin_TRANSPOSE,
            'MEAN':self.builtin_MEAN,
            'SUB':self.builtin_SUB,
            'DIV':self.builtin_DIV,
            'SQUEEZE':self.builtin_SQUEEZE,
            'UNIDIRECTIONAL_SEQUENCE_LSTM':self.builtin_UNIDIRECTIONAL_SEQUENCE_LSTM,
            'STRIDED_SLICE':self.builtin_STRIDED_SLICE,
            'BIDIRECTIONAL_SEQUENCE_RNN':self.builtin_BIDIRECTIONAL_SEQUENCE_RNN,
            'EXP':self.builtin_EXP,
            'TOPK_V2':self.builtin_TOPK_V2,
            'SPLIT':self.builtin_SPLIT,
            'SPLIT_V':self.builtin_SPLIT_V,
            'LOG_SOFTMAX':self.builtin_LOG_SOFTMAX,
            'DELEGATE':self.builtin_DELEGATE,
            'BIDIRECTIONAL_SEQUENCE_LSTM':self.builtin_BIDIRECTIONAL_SEQUENCE_LSTM,
            'CAST':self.builtin_CAST,
            'PRELU':self.builtin_PRELU,
            'MAXIMUM':self.builtin_MAXIMUM,
            'MINIMUM':self.builtin_MINIMUM,
            'LESS':self.builtin_LESS,
            'NEG':self.builtin_NEG,
            'PADV2':self.builtin_PADV2,
            'GREATER':self.builtin_GREATER,
            'GREATER_EQUAL':self.builtin_GREATER_EQUAL,
            'LESS_EQUAL':self.builtin_LESS_EQUAL,
            'SELECT':self.builtin_SELECT,
            'SLICE':self.builtin_SLICE,
            'SIN':self.builtin_SIN,
            'TRANSPOSE_CONV':self.builtin_TRANSPOSE_CONV,
            'SPARSE_TO_DENSE':self.builtin_SPARSE_TO_DENSE,
            'TILE':self.builtin_TILE,
            'EXPAND_DIMS':self.builtin_EXPAND_DIMS,
            'EQUAL':self.builtin_EQUAL,
            'NOT_EQUAL':self.builtin_NOT_EQUAL,
            'LOG':self.builtin_LOG,
            'SUM':self.builtin_SUM,
            'SQRT':self.builtin_SQRT,
            'RSQRT':self.builtin_RSQRT,
            'SHAPE':self.builtin_SHAPE,
            'POW':self.builtin_POW,
            'ARG_MIN':self.builtin_ARG_MIN,
            'FAKE_QUANT':self.builtin_FAKE_QUANT,
            'REDUCE_PROD':self.builtin_REDUCE_PROD,
            'REDUCE_MAX':self.builtin_REDUCE_MAX,
            'PACK':self.builtin_PACK,
            'LOGICAL_OR':self.builtin_LOGICAL_OR,
            'ONE_HOT':self.builtin_ONE_HOT,
            'LOGICAL_AND':self.builtin_LOGICAL_AND,
            'LOGICAL_NOT':self.builtin_LOGICAL_NOT,
            'UNPACK':self.builtin_UNPACK,
            'REDUCE_MIN':self.builtin_REDUCE_MIN,
            'FLOOR_DIV':self.builtin_FLOOR_DIV,
            'REDUCE_ANY':self.builtin_REDUCE_ANY,
            'SQUARE':self.builtin_SQUARE,
            'ZEROS_LIKE':self.builtin_ZEROS_LIKE,
            'FILL':self.builtin_FILL,
            'FLOOR_MOD':self.builtin_FLOOR_MOD,
            'RANGE':self.builtin_RANGE,
            'RESIZE_NEAREST_NEIGHBOR':self.builtin_RESIZE_NEAREST_NEIGHBOR,
            #post process
            'CUSTOM':self.builtin_CUSTOM,
         }


    def convert(self):
        transform = SGSModel_transform_onnx.TransformToSchema(self._net,self._netInputName,self._netInputShape,self._netOutputName,self._input_pack_model)
        self.net,self._maceOpArray = transform.run()
        self.getAllTensor(self._net)
        #self.creatOpNode(self._net)
        return self.convertToSGSModle(self._net)

    def getAllTensor(self, SGSMode):
        #1.collect all tensor (include constant and virable tensor)
        outputname_list = []
        outputShape_list = []
        #add constant tensor
        for tensor in SGSMode.tensors:
          self._tensorArray = np.append(self._tensorArray,tensor)
        #add op in tensor array
        for op in SGSMode.op:
          for outputTensor in op.output:
            outputname_list.append(outputTensor)
          for outputShape in op.output_shape:
            outputShape_list.append(outputShape.dims)
        #mace_check(len(outputname_list)==len(outputShape_list),
        #"output name num must same as shape num")
        for i in six.moves.range(len(outputname_list)):
          flag = False
          name = outputname_list[i]
          # ignor repeated tensor
          for tensor in self._tensorArray:
            if tensor.name == name:
              flag = True
          if flag:
            continue
          shape = outputShape_list[i]
          tensor = self.add_tensor(self._net, name, shape, mace_pb2.DT_FLOAT, None)
          self._tensorArray = np.append(self._tensorArray,tensor)

        #2.create a map form tensor name to tensor index
        #because buffer[0]==empty,so index start to 1
        for index,tensor in enumerate(self._tensorArray):
          name = tensor.name
          self._mapBufferToTensor[name]=index + 1

    def _creatBuffer(self):
        utility.annPrint("=== creat Buffer begin ===", fprint=True)
        if INVALID_FILTER:
           _SGSBufferArray = np.array([]) # all buffer array
           #the 0th entry of this array must be an empty buffer (sentinel).
           tflite.Buffer.BufferStartDataVector(self._builder,0)
           data = self._builder.EndVector(0)
           tflite.Buffer.BufferStart(self._builder)
           tflite.Buffer.BufferAddData(self._builder,data)
           firstEmptyBuffer = tflite.Buffer.BufferEnd(self._builder)
           _SGSBufferArray = np.append(_SGSBufferArray,firstEmptyBuffer)
           for tensor in self._tensorArray:
             _allByteArray = bytearray()
             tflite.Buffer.BufferStartDataVector(self._builder,0)
             data = self._builder.EndVector(0)
             tflite.Buffer.BufferStart(self._builder)
             tflite.Buffer.BufferAddData(self._builder,data)
             buffer = tflite.Buffer.BufferEnd(self._builder)
             #creat all tensor buffer
             _SGSBufferArray = np.append(_SGSBufferArray,buffer)#0.empty 1.filter 2.conv
        else:
            _SGSBufferArray = np.array([]) # all buffer array
            #the 0th entry of this array must be an empty buffer (sentinel).
            tflite.Buffer.BufferStartDataVector(self._builder,0)
            data = self._builder.EndVector(0)
            tflite.Buffer.BufferStart(self._builder)
            tflite.Buffer.BufferAddData(self._builder,data)
            firstEmptyBuffer = tflite.Buffer.BufferEnd(self._builder)
            _SGSBufferArray = np.append(_SGSBufferArray,firstEmptyBuffer)
            process = utility.ShowProcess(len(self._tensorArray))
            for tensor in self._tensorArray:
              _allByteArray = bytearray()
              if tensor.int32_data != []:
               ori_data = tensor.int32_data
              else:
               ori_data = tensor.float_data #NCHW
              ori_shape = tensor.dims
              _len = len(ori_data)
              data_format = tensor.data_format
              if _len != 0:
                if len(ori_shape) == 4 and data_format == mace_pb2.DT_NCHW:
                #transpose data to NHWC
                    utility.annPrint("Reshape ",tensor.name,"to NHWC")
                    data = np.array(ori_data)
                    data = data.reshape(ori_shape)
                    data = data.transpose(0,2,3,1)
                    data = list(data.flat)
                else:
                    data = ori_data
                utility.annPrint('set',tensor.name,'data to schema')
                data_len = len(data)
                if type(1)==type(data[0]):#data type is int
                  _allByteArray = struct.pack("i"*data_len, *data)
                else:#data type is floatpy
                  _allByteArray = struct.pack("f"*data_len, *data)
                  #_allByteArray.extend(_ubyteArray)
                tflite.Buffer.BufferStartDataVector(self._builder,len(_allByteArray))
                for meta_ubyte in reversed(_allByteArray):
                    self._builder.PrependByte(meta_ubyte)
                data = self._builder.EndVector(len(_allByteArray))#singal tensor buffer
              else:
                tflite.Buffer.BufferStartDataVector(self._builder,0)
                data = self._builder.EndVector(0)
              tflite.Buffer.BufferStart(self._builder)
              tflite.Buffer.BufferAddData(self._builder,data)
              buffer = tflite.Buffer.BufferEnd(self._builder)
              #creat all tensor buffer
              _SGSBufferArray = np.append(_SGSBufferArray,buffer)#0.empty 1.filter 2.conv
              process.show_process()
        utility.annPrint("=== creat Buffer end ===", fprint=True)
        return _SGSBufferArray

    def _creatTensor(self):
        utility.annPrint("=== creat Tensor begin ===",fprint=True)
        index = len(self._tensorArray) -1
        _SGSTensorArray = np.array([])
        for tensor in self._tensorArray:
          name = tensor.name
          tensor_name = self._builder.CreateString(name)
          data_type = tensor.data_type
          data_format = tensor.data_format
          shape = tensor.dims #NCHW
          #transpose shape to NHWC
          if len(shape) == 4 and data_format == mace_pb2.DT_NCHW:
            Transformer.transpose_shape(shape, [0, 2, 3, 1])
          tflite.Tensor.TensorStartShapeVector(self._builder,len(shape))
          for i in reversed(shape):
            self._builder.PrependInt32(i)
          shapes = self._builder.EndVector(len(shape))
          buffer_id = self._mapBufferToTensor[name]
          tflite.Tensor.TensorStart(self._builder)
          tflite.Tensor.TensorAddShape(self._builder,shapes)
          if data_type == mace_pb2.DT_INT32:
            tflite.Tensor.TensorAddType(self._builder,tflite.TensorType.TensorType().INT32)#2
          else:
            tflite.Tensor.TensorAddType(self._builder,tflite.TensorType.TensorType().FLOAT32)#0
          tflite.Tensor.TensorAddBuffer(self._builder,buffer_id)
          tflite.Tensor.TensorAddName(self._builder,tensor_name)
          tensor = tflite.Tensor.TensorEnd(self._builder)
          _SGSTensorArray = np.append(_SGSTensorArray,tensor)
          self._mapTensorToSubgraph[name] = index
          index -= 1
        utility.annPrint("=== creat Tensor end ===",fprint=True)
        return _SGSTensorArray

    def _creatOperator(self,SGSMode):
        utility.annPrint("=== creat Operator begin ===",fprint=True)
        _SGSOperatorArray = np.array([])
        data = None
        for op in reversed(self._maceOpArray) :
            type = op.type
            name = op.name
            #creat inputs
            input_name_array = op.input
            tflite.Operator.OperatorStartInputsVector(self._builder,len(input_name_array))
            for input_name in reversed(input_name_array):
              index = self._mapTensorToSubgraph[input_name]
              self._builder.PrependInt32(index)
            inputs =self._builder.EndVector(len(input_name_array))
            #creat outputs
            output_name_array = op.output
            tflite.Operator.OperatorStartOutputsVector(self._builder,len(output_name_array))
            for output_name in reversed(output_name_array):
              index = self._mapTensorToSubgraph[output_name]
              self._builder.PrependInt32(index)
            outputs =self._builder.EndVector(len(output_name_array))
            utility.annPrint('GEN',op.name,' builtin_op')
            builtinOpTypes,builtinOps = self._builtin_op[type](op)
            #find opcodeIndex
            opcodeIndex = 0
            for operatorCode in self._operatorCodeArray:
              if operatorCode == type:
                break
              else:
                opcodeIndex += 1

            if type == 'CUSTOM':
              cusByteArray = self.getDataFromFlexBuffer(op)
              tflite.Operator.OperatorStartCustomOptionsVector(self._builder,len(cusByteArray))
              for meta_ubyte in reversed(cusByteArray):
                self._builder.PrependByte(meta_ubyte)
              data = self._builder.EndVector(len(cusByteArray))#
            #creat Operator
            tflite.Operator.OperatorStart(self._builder)
            tflite.Operator.OperatorAddOpcodeIndex(self._builder,opcodeIndex)
            tflite.Operator.OperatorAddInputs(self._builder,inputs)
            tflite.Operator.OperatorAddOutputs(self._builder,outputs)
            if builtinOpTypes != None:
                tflite.Operator.OperatorAddBuiltinOptionsType(self._builder,builtinOpTypes)
                tflite.Operator.OperatorAddBuiltinOptions(self._builder,builtinOps)
            if data != None:
                tflite.Operator.OperatorAddCustomOptions(self._builder,data)
            operator = tflite.Operator.OperatorEnd(self._builder)
            data = None
            _SGSOperatorArray = np.append(_SGSOperatorArray,operator)
        utility.annPrint("=== creat Operator end ===",fprint=True)
        return _SGSOperatorArray

    def _creatSubGraph(self, SGSTensorArray, SGSOperatorArray):
        utility.annPrint("=== creat SubGraph begin ===",fprint=True)
        _SGSSubGraphArray = np.array([])
        graphName = self._builder.CreateString('eason')
        tflite.SubGraph.SubGraphStartTensorsVector(self._builder, len(SGSTensorArray))
        for SGSTensor in SGSTensorArray:
          self._builder.PrependUOffsetTRelative(int(SGSTensor))
        tensors = self._builder.EndVector(len(SGSTensorArray))
        #debug
        #tensors = 0
        tflite.SubGraph.SubGraphStartOperatorsVector(self._builder,len(SGSOperatorArray))
        for SGSOperator in SGSOperatorArray:
          self._builder.PrependUOffsetTRelative(int(SGSOperator))
        operators = self._builder.EndVector(len(SGSOperatorArray))
        #debug
        #operators = 0

        #set up input
        tflite.SubGraph.SubGraphStartInputsVector(self._builder,len(self._netInputName))
        for inputName in reversed(self._netInputName):
          inputIndex = self._mapTensorToSubgraph[inputName]
          self._builder.PrependInt32(inputIndex)
        inputIndexs = self._builder.EndVector(len(self._netInputName))


        #set up output
        tflite.SubGraph.SubGraphStartOutputsVector(self._builder,len(self._netOutputName))
        for outputName in reversed(self._netOutputName):
          outputIndex = self._mapTensorToSubgraph[outputName]
          self._builder.PrependInt32(outputIndex)
        outputIndexs = self._builder.EndVector(len(self._netOutputName))


        tflite.SubGraph.SubGraphStart(self._builder)
        tflite.SubGraph.SubGraphAddTensors(self._builder, tensors)
        tflite.SubGraph.SubGraphAddInputs(self._builder,inputIndexs)
        tflite.SubGraph.SubGraphAddOutputs(self._builder,outputIndexs)
        tflite.SubGraph.SubGraphAddOperators(self._builder,operators)
        tflite.SubGraph.SubGraphAddName(self._builder,graphName)
        subGraph = tflite.SubGraph.SubGraphEnd(self._builder)
        _SGSSubGraphArray = np.append(_SGSSubGraphArray,subGraph)
        utility.annPrint("=== creat SubGraph end ===",fprint=True)
        return _SGSSubGraphArray

    def _creatModel(self, SGSOperatorArray, SGSSubGraphArray, SGSBufferArray):
        utility.annPrint("=== creat Model begin ===",fprint=True)
        tflite.Model.ModelStartOperatorCodesVector(self._builder, len(self._operatorCodeArray))
        for operatorCode in reversed(self._operatorCodeArray):
          self._builder.PrependUOffsetTRelative(int(self._operatorCodeArray[operatorCode]))
        operatorCodes = self._builder.EndVector(len(self._operatorCodeArray))

        tflite.Model.ModelStartBuffersVector(self._builder, len(SGSBufferArray))
        for SGSBuffer in reversed(SGSBufferArray):
          self._builder.PrependUOffsetTRelative(int(SGSBuffer))
        buffers = self._builder.EndVector(len(SGSBufferArray))

        tflite.Model.ModelStartOperatorCodesVector(self._builder, len(SGSSubGraphArray))
        for SGSSubgraph in SGSSubGraphArray:
          self._builder.PrependUOffsetTRelative(int(SGSSubgraph))
        subgraphs = self._builder.EndVector(len(SGSSubGraphArray))
        #buffers = 0
        description = self._builder.CreateString('SGS MODEL')
        tflite.Model.ModelStart(self._builder)
        tflite.Model.ModelAddVersion(self._builder, 3)
        tflite.Model.ModelAddDescription(self._builder,description)
        tflite.Model.ModelAddOperatorCodes(self._builder,operatorCodes)
        tflite.Model.ModelAddSubgraphs(self._builder, subgraphs)
        tflite.Model.ModelAddBuffers(self._builder, buffers)
        model_b = tflite.Model.ModelEnd(self._builder)
        file_identifier = b'TFL3'
        self._builder.Finish(model_b, file_identifier)
        buf = self._builder.Output()
        utility.annPrint("=== creat Model end ===",fprint=True)
        return buf

    def _testModel(self, buf):
        model = tflite.Model.Model.GetRootAsModel(buf,0)
        assert model.Description().decode() == 'model test'
        assert model.Version() == 3
        ##   decode OperatorCodes   ##
        for i in six.moves.range(model.OperatorCodesLength()):
          #assert model.OperatorCodes(i).BuiltinCode() == tflite.BuiltinOperator.BuiltinOperator().CONV_2D #
          six.print_("model OperatorCodes",model.OperatorCodes(i).BuiltinCode())
        ##   decode Buffers   ##
        _testBufferArray = np.array([0,1,0,0])
        six.print_("model buffers len",model.BuffersLength())
        for i in six.moves.range(model.BuffersLength()):
          #assert model.Buffers(i).Data(0) == _testBufferArray[i]
          six.print_("model buffers",model.Buffers(i).DataAsNumpy())
        ##   decode Subgraph   ##
        subgraph = model.Subgraphs(0)
        for i in six.moves.range(subgraph.TensorsLength()):
          six.print_("===================================")
          six.print_("tensorName: ",subgraph.Tensors(i).Name().decode())
          six.print_("tensorShape: ",subgraph.Tensors(i).ShapeAsNumpy())
          six.print_("tensorType: ",subgraph.Tensors(i).Type())
          six.print_("tensorType: ",subgraph.Tensors(i).Buffer())
        six.print_("Map",self._mapBufferToTensor)
        six.print_("Map sub",self._mapTensorToSubgraph)
        six.print_("inputs: ",subgraph.InputsAsNumpy())
        six.print_("outputs: ",subgraph.OutputsAsNumpy())




    def convertToSGSModle(self, SGSMode):
        SGSBufferArray = self._creatBuffer()
        SGSTensorArray = self._creatTensor()
        SGSOperatorArray = self._creatOperator(SGSMode)
        SGSSubGraphArray = self._creatSubGraph(SGSTensorArray,SGSOperatorArray)
        buf = self._creatModel(SGSOperatorArray, SGSSubGraphArray, SGSBufferArray)
        return buf

    def creatBuiltinOp(self, op_tpye, op_arg):

        return builtinOpType,builtinOp
    def add_tensor(self, net, name, shape, data_type, value):
        tensor = net.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        tensor.float_data.extend(value)
        return tensor

    def getDataFromFlexBuffer(self,op):
        name = op.name
        so = ctypes.cdll.LoadLibrary
        if 'SGS_IPU_DIR' in os.environ:
          libSGS = 'libs/x86_64/libSGSCusOP.so'
          Project_path = os.environ['SGS_IPU_DIR']
        elif 'TOP_DIR' in os.environ:
          libSGS = '../SRC/Tool/libs/x86_64/libSGSCusOP.so'
          Project_path = os.environ['TOP_DIR']
        else:
          raise OSError('\033[31mRun `source cfg_env.sh` in Tool dir.\033[0m')
        libSGS_path = os.path.join(Project_path, libSGS)
        lib = so(libSGS_path)
        #start
        lib.startCreatFlexBuffer.restype = None
        lib.startCreatFlexBuffer()
        if name == 'TFLite_CaffeSSD_Detection_PostProcess':
            #set data
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_detections\0',10)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_classes_per_detection\0',1)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_score_threshold\0', 0.0)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_iou_threshold\0', 0.45)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_classes\0',20) #dont include background class

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'y_scale\0', 10.0)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'x_scale\0', 10.0)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'h_scale\0', 5.0)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'w_scale\0', 5.0)
        if name == 'TFLite_RFCSSD_Detection_PostProcess':
            #set data
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_detections\0',100)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_classes_per_detection\0',1)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_score_threshold\0', 0.4)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_iou_threshold\0', 0.45)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_classes\0',5) #dont include background class

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'y_scale\0', 10.0)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'x_scale\0', 10.0)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'h_scale\0', 5.0)

            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'w_scale\0', 5.0)
        elif name == 'TFLite_Detection_NMS':
            pass
        elif name == 'TFLite_YoloV2_Detection_PostProcess':
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'h_scale\0', 1.0)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'w_scale\0', 0.0)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_detections\0',100)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_classes_per_detection\0',1)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'side\0',13)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_classes\0',20)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_box\0',5)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'coords\0',4)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'confidence_threshold\0', 0.4)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_threshold\0', 0.45)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases0\0', 1.3221)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases1\0', 1.73145)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases2\0', 3.19275)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases3\0', 4.00944)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases4\0', 5.05587)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases5\0', 8.09892)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases6\0', 9.47112)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases7\0', 4.84053)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases8\0', 11.2364)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases9\0', 10.0071)
        elif name == 'TFLite_YoloV2_608_Detection_PostProcess':
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_detections\0',100)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_classes_per_detection\0',1)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'side\0',19)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_classes\0',80)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_box\0',5)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'coords\0',4)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'confidence_threshold\0', 0.4)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_threshold\0', 0.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases0\0', 0.57273)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases1\0', 0.677385)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases2\0', 1.87446)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases3\0', 2.06253)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases4\0', 3.33843)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases5\0', 5.47434)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases6\0', 7.88282)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases7\0', 3.52778)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases8\0', 9.77052)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases9\0', 9.16828)
        elif name == 'TFLite_YoloV3_Detection_PostProcess':
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_detections\0',100)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_classes_per_detection\0',1)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_size\0',3)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride0\0',32)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride1\0',16)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride2\0',8)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_box0\0',3)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_box1\0',3)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_box2\0',3)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_classes\0',80)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_score_threshold\0', 0.3)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_iou_threshold\0', 0.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box0_biases0\0', 116)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box0_biases1\0', 90)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box0_biases2\0', 156)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box0_biases3\0', 198)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box0_biases4\0', 373)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box0_biases5\0', 326)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box1_biases0\0', 30)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box1_biases1\0', 61)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box1_biases2\0', 62)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box1_biases3\0', 45)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box1_biases4\0', 59)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box1_biases5\0', 119)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box2_biases0\0', 10)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box2_biases1\0', 13)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box2_biases2\0', 16)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box2_biases3\0', 30)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box2_biases4\0', 33)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'num_box2_biases5\0', 23)
        elif name == 'TFLite_LanceNet_Detection_PostProcess':
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_detections\0',100)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_classes_per_detection\0',1)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'side\0',19)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_classes\0',1)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_box\0',9)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'coords\0',4)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'confidence_threshold\0', 0.3)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_threshold\0', 0.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases0\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases1\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases2\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases3\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases4\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases5\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases6\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases7\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases8\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases9\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases10\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases11\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases12\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases13\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases14\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases15\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases16\0', 9.5)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'biases17\0', 9.5)
        elif name == 'TFLite_FDA_Detection_PostProcess':
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_detections\0',100)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'max_classes_per_detection\0',1)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_classes\0',2)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_score_threshold\0', 0.05)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'nms_iou_threshold\0', 0.3)
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_variance\0', 3)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'variance0\0', 0.1)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'variance1\0', 0.2)
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'variance2\0', 0.2)
        elif name == 'SGS_LSTM':
            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            lib.insertKeyString(b'C0\0', b'add_3#output\0')
            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            lib.insertKeyString(b'SubNet\0', b'SGS_LSTM_sub1\0')

        #get cusData
        c_ubyte_p = POINTER(c_ubyte)
        lib.getFlexBufferData.restype = c_ubyte_p
        cusData = lib.getFlexBufferData()

        #get data len for loop
        lib.getFlexBufferLenth.restype = c_int
        bufferLen = lib.getFlexBufferLenth()

        #save cusData to bytearray
        _allByteArray = bytearray()
        for i in six.moves.range(bufferLen):
           _allByteArray.append(cusData[i])

        #end for release buffer
        lib.endCreatFlexBuffer.restype = None
        lib.endCreatFlexBuffer()
        return _allByteArray

    def builtin_ABS(self, op):
        #merge follow Activation op
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().AbsOptions
        #creat option
        tflite.AbsOptions.AbsOptionsStart(self._builder)
        builtinOp = tflite.AbsOptions.AbsOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().ABS)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'ABS' not in self._operatorCodeArray:
          self._operatorCodeArray['ABS'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_ADD(self, op):
        #merge follow Activation op
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().AddOptions
        #creat option
        tflite.AddOptions.AddOptionsStart(self._builder)
        tflite.AddOptions.AddOptionsAddFusedActivationFunction(self._builder,tflite.ActivationFunctionType.ActivationFunctionType().NONE)
        builtinOp = tflite.AddOptions.AddOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().ADD)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'ADD' not in self._operatorCodeArray:
          self._operatorCodeArray['ADD'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_CONCATENATION(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_axis_str:
            axis = arg[i].i
        #creat option
        tflite.ConcatenationOptions.ConcatenationOptionsStart(self._builder)
        tflite.ConcatenationOptions.ConcatenationOptionsAddAxis(self._builder,axis)
        builtinOp = tflite.ConcatenationOptions.ConcatenationOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().CONCATENATION)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'CONCATENATION' not in self._operatorCodeArray:
          self._operatorCodeArray['CONCATENATION'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_CONV_2D(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().Conv2DOptions
        arg = op.arg
        strideH,strideW = 0,0
        dilationH,dilationW = 1,1
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            #padding value should be divided by 2
            paddingL,paddingR,paddingT,paddingB = arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideH,strideW = arg[i].ints
          elif name == MaceKeyword.mace_dilations_str:
            dilationH,dilationW = arg[i].ints

        #creat option
        tflite.Conv2DOptions.Conv2DOptionsStart(self._builder)
        tflite.Conv2DOptions.Conv2DOptionsAddPadding(self._builder,tflite.Padding.Padding().CAFFE)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.Conv2DOptions.Conv2DOptionsAddStrideW(self._builder,strideW)
        tflite.Conv2DOptions.Conv2DOptionsAddStrideH(self._builder,strideH)
        tflite.Conv2DOptions.Conv2DOptionsAddDilationWFactor(self._builder,dilationW)
        tflite.Conv2DOptions.Conv2DOptionsAddDilationHFactor(self._builder,dilationH)
        builtinOp = tflite.Conv2DOptions.Conv2DOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().CONV_2D)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'CONV_2D' not in self._operatorCodeArray:
          self._operatorCodeArray['CONV_2D'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_DEPTHWISE_CONV_2D(self, op):
        arg = op.arg
        strideH,strideW = 0,0
        dilationH,dilationW = 1,1
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().DepthwiseConv2DOptions
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            #padding value should be divided by 2
            paddingL,paddingR,paddingT,paddingB = arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideH,strideW = arg[i].ints
          elif name == MaceKeyword.mace_dilations_str:
            dilationH,dilationW = arg[i].ints

        #creat option
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsStart(self._builder)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPadding(self._builder,tflite.Padding.Padding().CAFFE)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideW(self._builder,strideW)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideH(self._builder,strideH)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationWFactor(self._builder,dilationW)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationHFactor(self._builder,dilationH)
        builtinOp = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().DEPTHWISE_CONV_2D)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'DEPTHWISE_CONV_2D' not in self._operatorCodeArray:
          self._operatorCodeArray['DEPTHWISE_CONV_2D'] = operatorCode
        return builtinOpTypes,builtinOp


        pass
    def builtin_DEQUANTIZE(self, arg):


        pass
    def builtin_EMBEDDING_LOOKUP(self, arg):


        pass
    def builtin_FLOOR(self, arg):
        pass
    def builtin_ACTIVATION(self,op):
        pass
    def builtin_FULLY_CONNECTED(self, arg):
        builtinOp = None
        builtinOpTypes = None
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().FULLY_CONNECTED)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'FULLY_CONNECTED' not in self._operatorCodeArray:
          self._operatorCodeArray['FULLY_CONNECTED'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_HASHTABLE_LOOKUP(self, arg):


        pass
    def builtin_L2_NORMALIZATION(self, arg):


        pass
    def builtin_L2_POOL_2D(self, arg):


        pass
    def builtin_LOGISTIC(self, arg):
        builtinOp = None
        builtinOpTypes = None
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'LOGISTIC' not in self._operatorCodeArray:
          self._operatorCodeArray['LOGISTIC'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_LSH_PROJECTION(self, arg):


        pass
    def builtin_LSTM(self, arg):
        pass
    def builtin_AVERAGE_POOL_2D(self,op):
        arg = op.arg
        strideW,strideH,filter_width,filter_height,paddingW,paddingH= 0,0,0,0,0,0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().Pool2DOptions
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            paddingL,paddingR,paddingT,paddingB = arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideH,strideW = arg[i].ints
          elif name == MaceKeyword.mace_kernel_str:
            filter_height,filter_width= arg[i].ints
          elif name == MaceKeyword.mace_pooling_type_str:
            pooling_type = arg[i].i
        tflite.Pool2DOptions.Pool2DOptionsStart(self._builder)
        tflite.Pool2DOptions.Pool2DOptionsAddPadding(self._builder,tflite.Padding.Padding().CAFFE)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideW(self._builder,strideW)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideH(self._builder,strideH)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(self._builder,filter_width)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(self._builder,filter_height)
        builtinOp = tflite.Pool2DOptions.Pool2DOptionsEnd(self._builder)
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().AVERAGE_POOL_2D)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'AVERAGE_POOL_2D' not in self._operatorCodeArray:
          self._operatorCodeArray['AVERAGE_POOL_2D'] = operatorCode

        return builtinOpTypes,builtinOp

    def builtin_MAX_POOL_2D(self, op):
        arg = op.arg
        strideW,strideH,filter_width,filter_height= 0,0,0,0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().Pool2DOptions
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            paddingL,paddingR,paddingT,paddingB = arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideH,strideW = arg[i].ints
          elif name == MaceKeyword.mace_kernel_str:
            filter_height,filter_width = arg[i].ints
          elif name == MaceKeyword.mace_pooling_type_str:
            pooling_type = arg[i].i
        tflite.Pool2DOptions.Pool2DOptionsStart(self._builder)
        tflite.Pool2DOptions.Pool2DOptionsAddPadding(self._builder,tflite.Padding.Padding().CAFFE)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideW(self._builder,strideW)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideH(self._builder,strideH)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(self._builder,filter_width)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(self._builder,filter_height)
        builtinOp = tflite.Pool2DOptions.Pool2DOptionsEnd(self._builder)
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().MAX_POOL_2D)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'MAX_POOL_2D' not in self._operatorCodeArray:
          self._operatorCodeArray['MAX_POOL_2D'] = operatorCode
        return builtinOpTypes,builtinOp


    def builtin_MUL(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().MulOptions
        #creat option
        tflite.MulOptions.MulOptionsStart(self._builder)
        tflite.MulOptions.MulOptionsAddFusedActivationFunction(self._builder,tflite.ActivationFunctionType.ActivationFunctionType().NONE)
        builtinOp = tflite.MulOptions.MulOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().MUL)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'MUL' not in self._operatorCodeArray:
          self._operatorCodeArray['MUL'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RELU(self, op):
        builtinOp = None
        builtinOpTypes = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().RELU)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'RELU' not in self._operatorCodeArray:
          self._operatorCodeArray['RELU'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_LEAKY_RELU(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().LeakyReluOptions
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_activation_leakyrelu_coefficient_str:
            negative_slope = arg[i].f
        #creat option
        tflite.LeakyReluOptions.LeakyReluOptionsStart(self._builder)
        tflite.LeakyReluOptions.LeakyReluOptionsAddAlpha(self._builder,negative_slope)
        builtinOp = tflite.LeakyReluOptions.LeakyReluOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().LEAKY_RELU)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'LEAKY_RELU' not in self._operatorCodeArray:
          self._operatorCodeArray['LEAKY_RELU'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RELU6(self, op):
        builtinOp = None
        builtinOpTypes = None
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().RELU6)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'RELU6' not in self._operatorCodeArray:
          self._operatorCodeArray['RELU6'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RESHAPE(self, op):
        output_shape = op.output_shape[0].dims
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions
        #creat option
        tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(self._builder,len(output_shape))
        for i in reversed(output_shape):
            self._builder.PrependInt32(i)
        newShapes = self._builder.EndVector(len(output_shape))
        tflite.ReshapeOptions.ReshapeOptionsStart(self._builder)
        tflite.ReshapeOptions.ReshapeOptionsAddNewShape(self._builder,newShapes)
        builtinOp = tflite.ReshapeOptions.ReshapeOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'RESHAPE' not in self._operatorCodeArray:
          self._operatorCodeArray['RESHAPE'] = operatorCode
        return builtinOpTypes,builtinOp


    def builtin_RESIZE_BILINEAR(self, op):
        arg = op.arg
        align_corners = 0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ResizeBilinearOptions
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'align_corners':
            #padding value should be divided by 2
            align_corners = arg[i].i

        #creat option
        tflite.ResizeBilinearOptions.ResizeBilinearOptionsStart(self._builder)
        tflite.ResizeBilinearOptions.ResizeBilinearOptionsAddAlignCorners(self._builder,align_corners)
        builtinOp = tflite.ResizeBilinearOptions.ResizeBilinearOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().RESIZE_BILINEAR)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'RESIZE_BILINEAR' not in self._operatorCodeArray:
          self._operatorCodeArray['RESIZE_BILINEAR'] = operatorCode
        return builtinOpTypes,builtinOp


    def builtin_RESIZE_NEAREST_NEIGHBOR(self, op):
        arg = op.arg
        align_corners = 0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ResizeNearestNeighborOptions
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'align_corners':
            #padding value should be divided by 2
            align_corners = arg[i].i

        #creat option
        tflite.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsStart(self._builder)
        tflite.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsAddAlignCorners(self._builder,align_corners)
        builtinOp = tflite.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().RESIZE_NEAREST_NEIGHBOR)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'RESIZE_NEAREST_NEIGHBOR' not in self._operatorCodeArray:
          self._operatorCodeArray['RESIZE_NEAREST_NEIGHBOR'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RNN(self, op):
        pass

    def builtin_SOFTMAX(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SoftmaxOptions
        #creat option
        tflite.SoftmaxOptions.SoftmaxOptionsStart(self._builder)
        builtinOp = tflite.SoftmaxOptions.SoftmaxOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().SOFTMAX)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'SOFTMAX' not in self._operatorCodeArray:
          self._operatorCodeArray['SOFTMAX'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_SPACE_TO_DEPTH(self, op):
        pass
    def builtin_SVDF(self, op):
        pass
    def builtin_TANH(self, op):
        builtinOp = None
        builtinOpTypes = None
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().TANH)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'TANH' not in self._operatorCodeArray:
          self._operatorCodeArray['TANH'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_EMBEDDING_LOOKUP_SPARSE(self, op):


        pass
    def builtin_PAD(self, op):
        builtinOp = None
        builtinOpTypes = None
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().PAD)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'PAD' not in self._operatorCodeArray:
          self._operatorCodeArray['PAD'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_UNIDIRECTIONAL_SEQUENCE_RNN(self, op):


        pass
    def builtin_GATHER(self, op):


        pass
    def builtin_BATCH_TO_SPACE_ND(self, op):


        pass
    def builtin_SPACE_TO_BATCH_ND(self, op):


        pass
    def builtin_TRANSPOSE(self, op):
        builtinOp = None
        builtinOpTypes = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().TRANSPOSE)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'TRANSPOSE' not in self._operatorCodeArray:
          self._operatorCodeArray['TRANSPOSE'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_MEAN(self, op):
        builtinOpTypes = None
        builtinOp = None
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().MEAN)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'MEAN' not in self._operatorCodeArray:
          self._operatorCodeArray['MEAN'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_SUB(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SubOptions
        tflite.SubOptions.SubOptionsStart(self._builder)
        tflite.SubOptions.SubOptionsAddFusedActivationFunction(self._builder,tflite.ActivationFunctionType.ActivationFunctionType().NONE)
        builtinOp = tflite.SubOptions.SubOptionsEnd(self._builder)

        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().SUB)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'SUB' not in self._operatorCodeArray:
          self._operatorCodeArray['SUB'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_DIV(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().DivOptions
        #creat option
        tflite.DivOptions.DivOptionsStart(self._builder)
        tflite.DivOptions.DivOptionsAddFusedActivationFunction(self._builder,tflite.ActivationFunctionType.ActivationFunctionType().NONE)
        builtinOp = tflite.DivOptions.DivOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().DIV)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'DIV' not in self._operatorCodeArray:
          self._operatorCodeArray['DIV'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_SQUEEZE(self, op):


        pass
    def builtin_UNIDIRECTIONAL_SEQUENCE_LSTM(self, op):


        pass
    def builtin_STRIDED_SLICE(self, op):
        builtinOpTypes = None
        builtinOp = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().STRIDED_SLICE)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'STRIDED_SLICE' not in self._operatorCodeArray:
          self._operatorCodeArray['STRIDED_SLICE'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_BIDIRECTIONAL_SEQUENCE_RNN(self, op):


        pass
    def builtin_EXP(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ExpOptions
        tflite.ExpOptions.ExpOptionsStart(self._builder)
        builtinOp = tflite.ExpOptions.ExpOptionsEnd(self._builder)

        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().EXP)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'EXP' not in self._operatorCodeArray:
          self._operatorCodeArray['EXP'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_TOPK_V2(self, op):


        pass
    def builtin_SPLIT(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SplitOptions

        # create option
        numSplits = len(op.output_shape)
        tflite.SplitOptions.SplitOptionsStart(self._builder)
        tflite.SplitOptions.SplitOptionsAddNumSplits(self._builder, numSplits)
        builtinOp = tflite.SplitOptions.SplitOptionsEnd(self._builder)
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().SPLIT)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'SPLIT' not in self._operatorCodeArray:
          self._operatorCodeArray['SPLIT'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_SPLIT_V(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SplitVOptions

        # create option
        numSplits = len(op.output_shape)
        tflite.SplitVOptions.SplitVOptionsStart(self._builder)
        tflite.SplitVOptions.SplitVOptionsAddNumSplits(self._builder, numSplits)
        builtinOp = tflite.SplitVOptions.SplitVOptionsEnd(self._builder)
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().SPLIT_V)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'SPLIT_V' not in self._operatorCodeArray:
          self._operatorCodeArray['SPLIT_V'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_LOG_SOFTMAX(self, op):


        pass
    def builtin_DELEGATE(self, op):


        pass
    def builtin_BIDIRECTIONAL_SEQUENCE_LSTM(self, op):


        pass
    def builtin_CAST(self, op):


        pass
    def builtin_PRELU(self, op):
        builtinOpTypes = None
        builtinOp = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().PRELU)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'PRELU' not in self._operatorCodeArray:
          self._operatorCodeArray['PRELU'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_MAXIMUM(self, op):
        builtinOpTypes = None
        builtinOp = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().MAXIMUM)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'MAXIMUM' not in self._operatorCodeArray:
          self._operatorCodeArray['MAXIMUM'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_ARG_MAX(self, op):
        builtinOpTypes = None
        builtinOp = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().ARG_MAX)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'ARG_MAX' not in self._operatorCodeArray:
          self._operatorCodeArray['ARG_MAX'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_MINIMUM(self, op):
        builtinOpTypes = None
        builtinOp = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().MINIMUM)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'MINIMUM' not in self._operatorCodeArray:
          self._operatorCodeArray['MINIMUM'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_LESS(self, op):


        pass
    def builtin_NEG(self, op):


        pass
    def builtin_PADV2(self, op):


        pass
    def builtin_GREATER(self, op):


        pass
    def builtin_GREATER_EQUAL(self, op):
        builtinOpTypes = None
        builtinOp = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().GREATER_EQUAL)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'GREATER_EQUAL' not in self._operatorCodeArray:
          self._operatorCodeArray['GREATER_EQUAL'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_LESS_EQUAL(self, op):


        pass
    def builtin_SELECT(self, op):


        pass
    def builtin_SLICE(self, op):
        builtinOpTypes = None
        builtinOp = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().SLICE)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'SLICE' not in self._operatorCodeArray:
          self._operatorCodeArray['SLICE'] = operatorCode
        return builtinOpTypes, builtinOp


    def builtin_SIN(self, op):


        pass
    def builtin_TRANSPOSE_CONV(self, op):


        pass
    def builtin_SPARSE_TO_DENSE(self, op):


        pass
    def builtin_TILE(self, op):
        builtinOp = None
        builtinOpTypes = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().TILE)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'TILE' not in self._operatorCodeArray:
          self._operatorCodeArray['TILE'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_EXPAND_DIMS(self, op):


        pass
    def builtin_EQUAL(self, op):


        pass
    def builtin_NOT_EQUAL(self, op):

        pass
    def builtin_LOG(self, op):


        pass
    def builtin_SUM(self, op):
        builtinOp = None
        builtinOpTypes = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().SUM)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'SUM' not in self._operatorCodeArray:
          self._operatorCodeArray['SUM'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_SQRT(self, op):
        builtinOp = None
        builtinOpTypes = None
        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().SQRT)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'SQRT' not in self._operatorCodeArray:
          self._operatorCodeArray['SQRT'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RSQRT(self, op):


        pass
    def builtin_SHAPE(self, op):


        pass
    def builtin_POW(self, op):


        pass
    def builtin_ARG_MIN(self, op):


        pass
    def builtin_FAKE_QUANT(self, op):


        pass
    def builtin_REDUCE_PROD(self, op):


        pass
    def builtin_REDUCE_MAX(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ReducerOptions
        #creat option
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          keep_dim = 0
          if name == MaceKeyword.mace_keepdims_str:
            keep_dim = arg[i].i

        tflite.ReducerOptions.ReducerOptionsStart(self._builder)
        tflite.ReducerOptions.ReducerOptionsAddKeepDims(self._builder,keep_dim)
        builtinOp = tflite.ReducerOptions.ReducerOptionsEnd(self._builder)
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().REDUCE_MAX)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'REDUCE_MAX' not in self._operatorCodeArray:
          self._operatorCodeArray['REDUCE_MAX'] = operatorCode
        return builtinOpTypes,builtinOp


    def builtin_PACK(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().PackOptions
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_axis_str:
            axis = arg[i].i
          elif name == 'values_count':
            values_count = arg[i].i
        #creat option
        tflite.PackOptions.PackOptionsStart(self._builder)
        tflite.PackOptions.PackOptionsAddAxis(self._builder,axis)
        tflite.PackOptions.PackOptionsAddValuesCount(self._builder,values_count)
        builtinOp = tflite.PackOptions.PackOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().PACK)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'PACK' not in self._operatorCodeArray:
          self._operatorCodeArray['PACK'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_LOGICAL_OR(self, op):


        pass
    def builtin_ONE_HOT(self, op):


        pass
    def builtin_LOGICAL_AND(self, op):


        pass
    def builtin_LOGICAL_NOT(self, op):


        pass
    def builtin_UNPACK(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().UnpackOptions
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_axis_str:
            axis = arg[i].i
          elif name == 'num':
            num = arg[i].i
        #creat option
        tflite.PackOptions.UnpackOptionsStart(self._builder)
        tflite.PackOptions.UnpackOptionsAddAxis(self._builder,axis)
        tflite.PackOptions.UnpackOptionsAddNum(self._builder,num)
        builtinOp = tflite.PackOptions.UnpackOptionsEnd(self._builder)

        #creat Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().UNPACK)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'UNPACK' not in self._operatorCodeArray:
          self._operatorCodeArray['UNPACK'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_REDUCE_MIN(self, op):
        pass
    def builtin_FLOOR_DIV(self, op):
        pass
    def builtin_REDUCE_ANY(self, op):
        pass
    def builtin_SQUARE(self, arg):
        pass
    def builtin_ZEROS_LIKE(self, op):
        pass
    def builtin_FILL(self, op):
        pass
    def builtin_FLOOR_MOD(self, op):
        pass
    def builtin_RANGE(self, op):
        pass
    def builtin_CONCAT_EMBEDDINGS(self, op):
        pass
    def builtin_CALL(self, op):
        pass
    def builtin_CUSTOM(self, op):
        name = op.name
        if name == 'TFLite_CaffeSSD_Detection_PostProcess':#ssd postprocess
            #creat Operator Code
            customCode = self._builder.CreateString("TFLite_CaffeSSD_Detection_PostProcess")
        elif name == 'TFLite_RFCSSD_Detection_PostProcess':#rfc postprocess
            #creat Operator Code
            customCode = self._builder.CreateString("TFLite_RFCSSD_Detection_PostProcess")
        elif name == 'TFLite_Detection_NMS':
            #creat Operator Code
            customCode = self._builder.CreateString("TFLite_Detection_NMS")
        elif name == 'TFLite_YoloV2_Detection_PostProcess' or name == 'TFLite_YoloV2_608_Detection_PostProcess':
            customCode = self._builder.CreateString("TFLite_YoloV2_Detection_PostProcess")
        elif name == 'TFLite_YoloV3_Detection_PostProcess':
            customCode = self._builder.CreateString("TFLite_YoloV3_Detection_PostProcess")
        elif name == 'TFLite_LanceNet_Detection_PostProcess':
            customCode = self._builder.CreateString("TFLite_LanceNet_Detection_PostProcess")
        elif name == 'TFLite_FDA_Detection_PostProcess':
            customCode = self._builder.CreateString("TFLite_FDA_Detection_PostProcess")
        elif name == 'Dilation':
            customCode = self._builder.CreateString("Dilation")
        elif name == 'SGS_LSTM':
            customCode = self._builder.CreateString("SGS_LSTM")
        else:
            mace_check(0, "this Custom:%s layer will be support comming soon"%name)
        builtinOp = None
        builtinOpTypes = None
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().CUSTOM)
        tflite.OperatorCode.OperatorCodeAddCustomCode(self._builder,customCode)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'CUSTOM' not in self._operatorCodeArray:
          self._operatorCodeArray['CUSTOM'] = operatorCode
        return builtinOpTypes,builtinOp


def find_path(path, name):
    if path.split('/')[-1] == name:
        return path
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError('File `{}` not found in directory `{}`'.format(name, os.path.abspath(path)))
