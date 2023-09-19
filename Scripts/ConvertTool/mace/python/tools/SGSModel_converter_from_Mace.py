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
from mace.python.tools import SGSModel_check_tensor_shape_from_Mace

from mace.proto import mace_pb2
from third_party.python import flatbuffers
from third_party import tflite
import copy



INVALID_FILTER = False


class ConvertSGSModel(object):
    def __init__(self, net, inputName, outputName, outputDir, postnet_list=None, mode=None, convert_kb=None, inputNameShapeMap=None, input_name_format_map=None, input_pack_model=None, output_pack_model=None, platform=tflite):
        self._net = net
        self._platform = platform
        self._postnet_list = postnet_list
        self._mode = mode
        self._convert_kb = convert_kb
        self._netInputName = inputName
        self._netOutputName = outputName
        self._outputDir = outputDir
        self._netInputNameShapeMap = inputNameShapeMap
        self._input_pack_model = input_pack_model
        self._output_pack_model = output_pack_model
        self._input_name_format_map = input_name_format_map
        self._lstm_num = 0
        self._gru_num = 0
        self._mapBufferToTensor = {}
        self._mapTensorToSubgraph = {}
        self._netTensorNameShapeMap = {}
        self._netTensorNameElementMap = {}
        self._operatorCodeArray = collections.OrderedDict()
        self._builder = flatbuffers.Builder(1024)
        self._builtin_op = {
            'ABS':self.builtin_ABS,
            'ADD':self.builtin_ADD,
            'ADD_N':self.builtin_ADD_N,
            'ARG_MAX':self.builtin_ARG_MAX,
            'MAX_POOL_2D':self.builtin_MAX_POOL_2D,
            'AVERAGE_POOL_2D':self.builtin_AVERAGE_POOL_2D,
            'CONCATENATION':self.builtin_CONCATENATION,
            'CONV_2D':self.builtin_CONV_2D,
            'CONV_3D':self.builtin_CONV_3D,
            'COS':self.builtin_COS,
            'DEPTHWISE_CONV_2D':self.builtin_DEPTHWISE_CONV_2D,
            'DEQUANTIZE':self.builtin_DEQUANTIZE,
            'EMBEDDING_LOOKUP':self.builtin_EMBEDDING_LOOKUP,
            'FLOOR':self.builtin_FLOOR,
            'FLOOR_DIV':self.builtin_FLOOR_DIV,
            'Activation':self.builtin_ACTIVATION,
            'FULLY_CONNECTED':self.builtin_FULLY_CONNECTED,
            'HASHTABLE_LOOKUP':self.builtin_HASHTABLE_LOOKUP,
            'L2_NORMALIZATION':self.builtin_L2_NORMALIZATION,
            'L2_POOL_2D':self.builtin_L2_POOL_2D,
            'LOGISTIC':self.builtin_LOGISTIC,
            'LSH_PROJECTION':self.builtin_LSH_PROJECTION,
            'LSTM':self.builtin_LSTM,
            'GRU':self.builtin_GRU,
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
            'TANH':self.builtin_TANH,
            'CONCAT_EMBEDDINGS':self.builtin_CONCAT_EMBEDDINGS,
            'CALL':self.builtin_CALL,
            'EMBEDDING_LOOKUP_SPARSE':self.builtin_EMBEDDING_LOOKUP_SPARSE,
            'PAD':self.builtin_PAD,
            'MIRROR_PAD':self.builtin_MIRROR_PAD,
            'UNIDIRECTIONAL_SEQUENCE_RNN':self.builtin_UNIDIRECTIONAL_SEQUENCE_RNN,
            'GATHER':self.builtin_GATHER,
            'Greater':self.builtin_GREATER,
            'BATCH_TO_SPACE_ND':self.builtin_BATCH_TO_SPACE_ND,
            'SPACE_TO_BATCH_ND':self.builtin_SPACE_TO_BATCH_ND,
            'TRANSPOSE':self.builtin_TRANSPOSE,
            'MEAN':self.builtin_MEAN,
            'SUB':self.builtin_SUB,
            'DIV':self.builtin_DIV,
            'SQUEEZE':self.builtin_SQUEEZE,
            'UNIDIRECTIONAL_SEQUENCE_LSTM':self.builtin_UNIDIRECTIONAL_SEQUENCE_LSTM,
            'STRIDED_SLICE':self.builtin_STRIDEDSLICE,
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
            'REDUCE_ANY':self.builtin_REDUCE_ANY,
            'SQUARE':self.builtin_SQUARE,
            'ZEROS_LIKE':self.builtin_ZEROS_LIKE,
            'FILL':self.builtin_FILL,
            'FLOOR_MOD':self.builtin_FLOOR_MOD,
            'RANGE':self.builtin_RANGE,
            'RESIZE_NEAREST_NEIGHBOR':self.builtin_RESIZE_NEAREST_NEIGHBOR,
            'ELU':self.builtin_ELU,
            'BATCH_MATMUL':self.builtin_BATCHMATMUL,
            'ROUND':self.builtin_ROUND,
            'CUSTOM':self.builtin_CUSTOM,
         }

    def convert(self):
        return self.convertToSGSModel(self._net,self._postnet_list, self._mode)


    def getAllTensor(self, Model, mode=None, postmodel_list=None):
        #1.collect all tensor (include constant and virable tensor)
        outputname_list = []
        outputShape_list = []
        tensorArray= np.array([])
        mapBufferToTensor = {}
        #add constant tensor
        for tensor in Model.tensors:
          tensorArray = np.append(tensorArray,tensor)
        tensorArray = tensorArray.tolist()
        tensorArrayIndex = copy.deepcopy(tensorArray)
        if mode == 'append':
            for tensor in tensorArrayIndex:
                for j in range(len(Model.output_info)):
                    if tensor.name == Model.output_info[j].name:
                        tensorArray.remove(tensor)
        tensorArray = np.array(tensorArray)
        if postmodel_list is not None:
            for i in range(len(postmodel_list)):
                for tensor in postmodel_list[i].tensors:
                    tensorArray = np.append(tensorArray,tensor)
        #add op in tensor array
        if mode == 'append':
            for op in postmodel_list[0].op:
                for outputTensor in op.output:
                    outputname_list.append(outputTensor)
                for outputShape in op.output_shape:
                    outputShape_list.append(outputShape.dims)
        else:
            for op in Model.op:
              for outputTensor in op.output:
                  outputname_list.append(outputTensor)
              for outputShape in op.output_shape:
                  outputShape_list.append(outputShape.dims)

        #mace_check(len(outputname_list)==len(outputShape_list),
        #"output name num must same as shape num")
        for i in six.moves.range(len(outputname_list)):
          flag = False
          name = outputname_list[i]

          # ignore repeated tensor
          for tensor in tensorArray:
            if tensor.name == name:
              flag = True
          if flag:
            continue

          shape = outputShape_list[i]
          tensor = self.add_tensor(self._net, name, shape, mace_pb2.DT_FLOAT, None)
          tensorArray = np.append(tensorArray,tensor)

        #2.create a map from tensor name to tensor index
        #because buffer[0]==empty,so index start to 1
        for index,tensor in enumerate(tensorArray):
          name = tensor.name
          shape = tensor.dims
          self._netTensorNameShapeMap[name] = shape
          if tensor.int32_data != None:
            value = tensor.int32_data
            self._netTensorNameElementMap[name] = value
          mapBufferToTensor[name]=index + 1
        return tensorArray,mapBufferToTensor

    def _createOpcodeByIPUversion(self, E_opType):
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        if getIPUVersion() == 'M6' or getIPUVersion() == 'I6E':
            if E_opType > 127:#schema 1.14 The maximum number of operators supported is 128(0~127)
                mace_check(0, 'Model schema version is higher than 1.14, Please check model !! \n ')
            tflite.OperatorCode.OperatorCodeAddDeprecatedBuiltinCode(self._builder, E_opType)
        else:
            tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, E_opType)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        return operatorCode

    def _createBuffer(self, tensorArray):
        utility.annPrint("=== create Buffer begin ===", fprint=True)
        if INVALID_FILTER:
           _SGSBufferArray = np.array([]) # all buffer array
           #the 0th entry of this array must be an empty buffer (sentinel).
           tflite.Buffer.BufferStartDataVector(self._builder,0)
           data = self._builder.EndVector(0)
           tflite.Buffer.BufferStart(self._builder)
           tflite.Buffer.BufferAddData(self._builder,data)
           firstEmptyBuffer = tflite.Buffer.BufferEnd(self._builder)
           _SGSBufferArray = np.append(_SGSBufferArray,firstEmptyBuffer)
           for tensor in tensorArray:
             _allByteArray = bytearray()
             tflite.Buffer.BufferStartDataVector(self._builder,0)
             data = self._builder.EndVector(0)
             tflite.Buffer.BufferStart(self._builder)
             tflite.Buffer.BufferAddData(self._builder,data)
             buffer = tflite.Buffer.BufferEnd(self._builder)
             #create all tensor buffer
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
            process = utility.ShowProcess(len(tensorArray))
            tensor_name =[]
            for tensor in tensorArray:
              tensor_name.append(tensor.name)
            for tensor in tensorArray:
              _allByteArray = bytearray()
              if tensor.int32_data != []:
               ori_data = tensor.int32_data
              else:
               ori_data = tensor.float_data
              ori_shape = tensor.dims
              _len = len(ori_data)
              data_format = tensor.data_format
              if _len != 0:
                if len(ori_shape) == 4 and data_format == mace_pb2.DT_NCHW and self._platform != 'tflite':
                #transpose ONNX data format to NHWC
                    if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
                        utility.annPrint("Reshape ",tensor.name,"to NHWC")
                        data = np.array(ori_data)
                        data = data.reshape(ori_shape)
                        data = data.transpose(0,2,3,1)
                        data = data.flatten()
                    else:
                        data = ori_data
                    self._netTensorNameShapeMap[tensor.name] = tensor.dims
                    if tensor.int32_data != None:
                      value = tensor.int32_data
                      self._netTensorNameElementMap[tensor.name] = value
                else:
                    data = ori_data
                utility.annPrint('set',tensor.name,'data to schema')
                data_len = len(data)
                data_type = tensor.data_type
                if data_type == mace_pb2.DT_INT32:#data type is int32
                  _allByteArray = struct.pack("i"*data_len, *data)
                elif data_type == mace_pb2.DT_INT8:#data type is int8
                  _allByteArray = struct.pack("b"*data_len, *data)
                elif data_type == mace_pb2.DT_UINT8:#data type is uint8
                  _allByteArray = struct.pack("B"*data_len, *data)
                else:#data type is float
                  _allByteArray = struct.pack("f"*data_len, *data)
                data = self._builder.CreateByteVector(_allByteArray)
              else:
                tflite.Buffer.BufferStartDataVector(self._builder,0)
                data = self._builder.EndVector(0)
              tflite.Buffer.BufferStart(self._builder)
              tflite.Buffer.BufferAddData(self._builder,data)
              buffer = tflite.Buffer.BufferEnd(self._builder)
              #create all tensor buffer
              _SGSBufferArray = np.append(_SGSBufferArray,buffer)#0.empty 1.filter 2.conv
              process.show_process()
        utility.annPrint("=== create Buffer end ===", fprint=True)
        return _SGSBufferArray

    def _createTensor(self,tensorArray,mapBufferToTensor):
        utility.annPrint("=== create Tensor begin ===",fprint=True)
        index = len(tensorArray) -1
        _SGSTensorArray = np.array([])
        mapTensorToSubgraph = {}
        for tensor in tensorArray:
          name = tensor.name
          tensor_name = self._builder.CreateString(name)
          data_type = tensor.data_type
          data_format = tensor.data_format
          shape = tensor.dims
          bQuantized = tensor.quantized
          maxval = tensor.maxval
          minval = tensor.minval
          scale = tensor.scale
          zero_point = tensor.zero_point
          sparsity = tensor.sparsity
          shape_signature = tensor.shape_signature
          #transpose ONNX tensor shape format to NHWC
          if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
              if len(shape) == 4 and data_format == mace_pb2.DT_NCHW and self._platform != 'tflite':
                  Transformer.transpose_shape(shape, [0, 2, 3, 1])
                  self._netTensorNameShapeMap[tensor.name] = shape
                  if tensor.int32_data != None:
                    value = tensor.int32_data
                    self._netTensorNameElementMap[tensor.name] = value
          ####ADD QUANTIZATION
          #min
          tflite.QuantizationParameters.StartMinVector(self._builder,len(minval))
          for i in reversed(minval):
            self._builder.PrependFloat32(i)
          minvals_offset = self._builder.EndVector(len(minval))
          #max
          tflite.QuantizationParameters.StartMaxVector(self._builder,len(maxval))
          for i in reversed(maxval):
            self._builder.PrependFloat32(i)
          maxvals_offset = self._builder.EndVector(len(maxval))
          #scale
          tflite.QuantizationParameters.StartScaleVector(self._builder,len(scale))
          for i in reversed(scale):
            self._builder.PrependFloat32(i)
          scales_offset = self._builder.EndVector(len(scale))
          #zero_point
          tflite.QuantizationParameters.StartZeroPointVector(self._builder,len(zero_point))
          for i in reversed(zero_point):
            self._builder.PrependInt64(i)
          zero_points_offset = self._builder.EndVector(len(zero_point))
          tflite.QuantizationParameters.Start(self._builder)
          tflite.QuantizationParameters.AddMin(self._builder, minvals_offset)
          tflite.QuantizationParameters.AddMax(self._builder, maxvals_offset)
          tflite.QuantizationParameters.AddScale(self._builder, scales_offset)
          tflite.QuantizationParameters.AddZeroPoint(self._builder, zero_points_offset)
          Quantization = tflite.QuantizationParameters.End(self._builder)

          ####ADD SPARSITY
          #traversal_order
          tflite.SparsityParameters.StartTraversalOrderVector(self._builder,len(sparsity.traversal_order))
          for i in reversed(sparsity.traversal_order):
            self._builder.PrependInt32(i)
          traversal_order_offset = self._builder.EndVector(len(sparsity.traversal_order))
          #block_map
          tflite.SparsityParameters.StartBlockMapVector(self._builder,len(sparsity.block_map))
          for i in reversed(sparsity.block_map):
            self._builder.PrependInt32(i)
          block_map_offset = self._builder.EndVector(len(sparsity.block_map))
          #dim_metadata
          _DimMetadataOffset_array = np.array([])
          for i in range(len(sparsity.dim_metadata)):
            dim_metadata = sparsity.dim_metadata[i]
            dense_size = dim_metadata.dense_size
            E_ArrayIndicesVecType = 0
            E_ArraySegmentsVecType = 0
            _indicesVector_offset = 0
            _segmentsVector_offset = 0
            E_format = dim_metadata.format


            #add ArrayIndices
            if E_format == 1:# SPARSE_CSR
                array_indices = dim_metadata.array_indices
                array_segments = dim_metadata.array_segments
                ArrayIndicesVecType = dim_metadata.ArrayIndicesVecType
                ArraySegmentsVecType = dim_metadata.ArraySegmentsVecType
                E_ArrayIndicesVecType = self.tflite_getEnum(tflite.SparseIndexVector.SparseIndexVector, ArrayIndicesVecType)
                E_ArraySegmentsVecType = self.tflite_getEnum(tflite.SparseIndexVector.SparseIndexVector, ArraySegmentsVecType)
                if ArrayIndicesVecType == 'Int32Vector':
                    tflite.Int32Vector.StartValuesVector(self._builder,len(array_indices.s32Vector.values))
                    for i in reversed(array_indices.s32Vector.values):
                        self._builder.PrependInt32(i)
                    Vector_offset = self._builder.EndVector(len(array_indices.s32Vector.values))
                    tflite.Int32Vector.Start(self._builder)
                    tflite.Int32Vector.AddValues(self._builder, Vector_offset)
                    _indicesVector_offset = tflite.Int32Vector.End(self._builder)
                elif ArrayIndicesVecType == 'Uint16Vector':
                    tflite.Uint16Vector.StartValuesVector(self._builder,len(array_indices.u16Vector.values))
                    for i in reversed(array_indices.u16Vector.values):
                        self._builder.PrependUint16(i)
                    Vector_offset = self._builder.EndVector(len(array_indices.u16Vector.values))
                    tflite.Uint16Vector.Start(self._builder)
                    tflite.Uint16Vector.AddValues(self._builder, Vector_offset)
                    _indicesVector_offset = tflite.Uint16Vector.End(self._builder)
                elif ArrayIndicesVecType == 'Uint8Vector':
                    tflite.Uint8Vector.StartValuesVector(self._builder,len(array_indices.u8Vector.values))
                    for i in reversed(array_indices.u8Vector.values):
                        self._builder.PrependUint8(i)
                    Vector_offset = self._builder.EndVector(len(array_indices.u8Vector.values))
                    tflite.Uint8Vector.Start(self._builder)
                    tflite.Uint8Vector.AddValues(self._builder, Vector_offset)
                    _indicesVector_offset = tflite.Uint8Vector.End(self._builder)
                #add ArraySegments
                if ArraySegmentsVecType == 'Int32Vector':
                    tflite.Int32Vector.StartValuesVector(self._builder,len(array_segments.s32Vector.values))
                    for i in reversed(array_segments.s32Vector.values):
                        self._builder.PrependInt32(i)
                    Vector_offset = self._builder.EndVector(len(array_segments.s32Vector.values))
                    tflite.Int32Vector.Start(self._builder)
                    tflite.Int32Vector.AddValues(self._builder, Vector_offset)
                    _segmentsVector_offset = tflite.Int32Vector.End(self._builder)
                elif ArraySegmentsVecType == 'Uint16Vector':
                    tflite.Uint16Vector.StartValuesVector(self._builder,len(array_segments.u16Vector.values))
                    for i in reversed(array_segments.u16Vector.values):
                        self._builder.PrependUint16(i)
                    Vector_offset = self._builder.EndVector(len(array_segments.u16Vector.values))
                    tflite.Uint16Vector.Start(self._builder)
                    tflite.Uint16Vector.AddValues(self._builder, Vector_offset)
                    _segmentsVector_offset = tflite.Uint16Vector.End(self._builder)
                elif ArraySegmentsVecType == 'Uint8Vector':
                    tflite.Uint8Vector.StartValuesVector(self._builder,len(array_segments.u8Vector.values))
                    for i in reversed(array_segments.u8Vector.values):
                        self._builder.PrependUint8(i)
                    Vector_offset = self._builder.EndVector(len(array_segments.u8Vector.values))
                    tflite.Uint8Vector.Start(self._builder)
                    tflite.Uint8Vector.AddValues(self._builder, Vector_offset)
                    _segmentsVector_offset = tflite.Uint8Vector.End(self._builder)
            tflite.DimensionMetadata.Start(self._builder)
            tflite.DimensionMetadata.AddDenseSize(self._builder, dense_size)
            tflite.DimensionMetadata.AddFormat(self._builder, E_format)
            tflite.DimensionMetadata.AddArrayIndices(self._builder, _indicesVector_offset)
            tflite.DimensionMetadata.AddArrayIndicesType(self._builder, E_ArrayIndicesVecType)
            tflite.DimensionMetadata.AddArraySegments(self._builder, _segmentsVector_offset)
            tflite.DimensionMetadata.AddArraySegmentsType(self._builder, E_ArraySegmentsVecType)
            DimensionMetadata_offset = tflite.DimensionMetadata.End(self._builder)
            _DimMetadataOffset_array = np.append(_DimMetadataOffset_array,DimensionMetadata_offset)

          tflite.SparsityParameters.StartDimMetadataVector(self._builder,len(sparsity.dim_metadata))
          for offset_iter in _DimMetadataOffset_array:
            self._builder.PrependUOffsetTRelative(int(offset_iter))
          dim_metadata_offset = self._builder.EndVector(len(sparsity.dim_metadata))

          tflite.SparsityParameters.Start(self._builder)
          tflite.SparsityParameters.AddTraversalOrder(self._builder, traversal_order_offset)
          tflite.SparsityParameters.AddBlockMap(self._builder, block_map_offset)
          tflite.SparsityParameters.AddDimMetadata(self._builder, dim_metadata_offset)
          sparsity = tflite.SparsityParameters.End(self._builder)

          ####ADD SHAPE_SIGNATURE
          tflite.Tensor.TensorStartShapeSignatureVector(self._builder,len(shape_signature))
          for i in reversed(shape_signature):
            self._builder.PrependInt32(i)
          shape_signatures = self._builder.EndVector(len(shape_signature))
          ####ADD SHAPE
          tflite.Tensor.TensorStartShapeVector(self._builder,len(shape))
          for i in reversed(shape):
            self._builder.PrependInt32(i)
          shapes = self._builder.EndVector(len(shape))
          buffer_id = mapBufferToTensor[name]
          tflite.Tensor.TensorStart(self._builder)
          tflite.Tensor.TensorAddShape(self._builder,shapes)
          if data_type == mace_pb2.DT_INT32:
            tflite.Tensor.TensorAddType(self._builder,tflite.TensorType.TensorType().INT32)#2
          elif data_type == mace_pb2.DT_INT8:
            tflite.Tensor.TensorAddType(self._builder,tflite.TensorType.TensorType().INT8)
          elif data_type == mace_pb2.DT_UINT8:
            tflite.Tensor.TensorAddType(self._builder,tflite.TensorType.TensorType().UINT8)
          else:
            tflite.Tensor.TensorAddType(self._builder,tflite.TensorType.TensorType().FLOAT32)#0
          tflite.Tensor.TensorAddBuffer(self._builder,buffer_id)
          tflite.Tensor.TensorAddName(self._builder,tensor_name)
          #update schema2.7
          tflite.Tensor.TensorAddQuantization(self._builder, Quantization)
          tflite.Tensor.TensorAddSparsity(self._builder, sparsity)
          tflite.Tensor.TensorAddShapeSignature(self._builder, shape_signatures)
          tensor = tflite.Tensor.TensorEnd(self._builder)
          _SGSTensorArray = np.append(_SGSTensorArray,tensor)
          mapTensorToSubgraph[name] = index
          index -= 1

        utility.annPrint("=== create Tensor end ===",fprint=True)
        return _SGSTensorArray,mapTensorToSubgraph

    def _createOperator(self,SGSMode,maceOpArray,mapTensorToSubgraph):
        utility.annPrint("=== create Operator begin ===",fprint=True)
        _SGSOperatorArray = np.array([])
        data = None
        for op in reversed(maceOpArray) :
            type = op.type
            name = op.name
            #create inputs
            input_name_array = op.input
            tflite.Operator.OperatorStartInputsVector(self._builder,len(input_name_array))
            for input_name in reversed(input_name_array):
              index = mapTensorToSubgraph[input_name]
              self._builder.PrependInt32(index)
            inputs =self._builder.EndVector(len(input_name_array))
            #create outputs
            output_name_array = op.output
            tflite.Operator.OperatorStartOutputsVector(self._builder,len(output_name_array))
            for output_name in reversed(output_name_array):
              index = mapTensorToSubgraph[output_name]
              self._builder.PrependInt32(index)
            outputs =self._builder.EndVector(len(output_name_array))
            utility.annPrint('GEN',op.name,' builtin_op')

            #find opcodeIndex
            opcodeIndex = 0
            if type != 'CUSTOM':
              builtinOpTypes,builtinOps = self._builtin_op[type](op)
              for operatorCode in self._operatorCodeArray:
                if operatorCode == type:
                  break
                else:
                  opcodeIndex += 1
            else:
              builtinOpTypes,builtinOps,customName = self._builtin_op[type](op)
              for operatorCode in self._operatorCodeArray:
                if operatorCode == type + customName:
                  break
                else:
                  opcodeIndex += 1

            if type == 'CUSTOM' or type == 'ELU':
              cusByteArray = self.getDataFromFlexBuffer(op)
              tflite.Operator.OperatorStartCustomOptionsVector(self._builder,len(cusByteArray))
              for meta_ubyte in reversed(cusByteArray):
                self._builder.PrependByte(meta_ubyte)
              data = self._builder.EndVector(len(cusByteArray))
            #ADD Intermediates
            intermediates_array = op.intermediates
            tflite.Operator.OperatorStartIntermediatesVector(self._builder,len(intermediates_array))
            for i in reversed(intermediates_array):
                self._builder.PrependInt32(i)
            intermediates = self._builder.EndVector(len(intermediates_array))
            #create Operator
            tflite.Operator.OperatorStart(self._builder)
            tflite.Operator.OperatorAddOpcodeIndex(self._builder,opcodeIndex)
            tflite.Operator.OperatorAddInputs(self._builder,inputs)
            tflite.Operator.OperatorAddOutputs(self._builder,outputs)
            tflite.Operator.OperatorAddIntermediates(self._builder,intermediates)
            if builtinOpTypes != None:
                tflite.Operator.OperatorAddBuiltinOptionsType(self._builder,builtinOpTypes)
                tflite.Operator.OperatorAddBuiltinOptions(self._builder,builtinOps)
            if data != None:
                tflite.Operator.OperatorAddCustomOptions(self._builder,data)
            operator = tflite.Operator.OperatorEnd(self._builder)
            data = None
            _SGSOperatorArray = np.append(_SGSOperatorArray,operator)
        utility.annPrint("=== create Operator end ===",fprint=True)
        return _SGSOperatorArray

    def _createSubGraph(self, SGSTensorArray, SGSOperatorArray,subgraphname, netInputName,netOutputName,mapTensorToSubgraph):
        utility.annPrint("=== create SubGraph begin ===",fprint=True)
        _SGSSubGraphArray = np.array([])
        graphName = self._builder.CreateString(subgraphname)
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
        tflite.SubGraph.SubGraphStartInputsVector(self._builder,len(netInputName))
        for inputName in reversed(netInputName):
          inputIndex = mapTensorToSubgraph[inputName]
          self._builder.PrependInt32(inputIndex)
        inputIndexs = self._builder.EndVector(len(netInputName))


        #set up output
        tflite.SubGraph.SubGraphStartOutputsVector(self._builder,len(netOutputName))
        for outputName in reversed(netOutputName):
          outputIndex = mapTensorToSubgraph[outputName]
          self._builder.PrependInt32(outputIndex)
        outputIndexs = self._builder.EndVector(len(netOutputName))


        tflite.SubGraph.SubGraphStart(self._builder)
        tflite.SubGraph.SubGraphAddTensors(self._builder, tensors)
        tflite.SubGraph.SubGraphAddInputs(self._builder,inputIndexs)
        tflite.SubGraph.SubGraphAddOutputs(self._builder,outputIndexs)
        tflite.SubGraph.SubGraphAddOperators(self._builder,operators)
        tflite.SubGraph.SubGraphAddName(self._builder,graphName)
        subGraph = tflite.SubGraph.SubGraphEnd(self._builder)
        _SGSSubGraphArray = np.append(_SGSSubGraphArray,subGraph)
        utility.annPrint("=== create SubGraph end ===",fprint=True)
        return _SGSSubGraphArray

    def _createModel(self,    SGSSubGraphArray, SGSBufferArray):
        utility.annPrint("=== create Model begin ===",fprint=True)
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
        description = self._builder.CreateString('Original Float')
        tflite.Model.ModelStart(self._builder)
        tflite.Model.ModelAddVersion(self._builder, 3)
        tflite.Model.ModelAddDescription(self._builder,description)
        tflite.Model.ModelAddOperatorCodes(self._builder,operatorCodes)
        tflite.Model.ModelAddSubgraphs(self._builder, subgraphs)
        tflite.Model.ModelAddBuffers(self._builder, buffers)
        model_b = tflite.Model.ModelEnd(self._builder)
        if getIPUVersion() == 'M6' or getIPUVersion() == 'I6E':
            file_identifier = b'TFL3'
        else:
            file_identifier = b'SIM2'
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




    def convertToSGSModel(self, Model, PostModel_list, mode):

        tensorArray= np.array([])
        mapBufferToTensor = {}
        mapBufferToTensor1 = {}
        mapTensorToSubgraph = {}
        maceOpArray = np.array([])
        maceOpArray1 = np.array([])
        SGSTensorArray = np.array([])
        SGSBufferArray_total = np.array([])
        SubGraphArray_total = np.array([])

        if mode == 'append':
            if len(Model.output_info) != len(PostModel_list[0].input_info):
                six.print_("model output and postmodel input must be same")
                assert 0
            else:
                for i in range(len(Model.output_info)):
                    if Model.output_info[i] != PostModel_list[0].input_info[i]:
                        six.print_("model output and postmodel input must be same")
                        assert 0
            # create op
            transform = SGSModel_transform_tflite.TransformToSchema(Model)
            Model,maceOpArray = transform.run()
            transform1 = SGSModel_transform_tflite.TransformToSchema(PostModel_list[0])
            PostModel_list[0],maceOpArray1 = transform1.run()
            for op  in maceOpArray1:
                maceOpArray = np.append(maceOpArray,op)
            tensorArray, mapBufferToTensor = self.getAllTensor(Model,mode,PostModel_list)
            SGSBufferArray = self._createBuffer(tensorArray)
            SGSTensorArray,mapTensorToSubgraph = self._createTensor(tensorArray, mapBufferToTensor)
            SGSOperatorArray = self._createOperator(Model,maceOpArray,mapTensorToSubgraph)
            subgraphname = Model.arg[0].str
            netInputName = []
            netOutputName = []
            netInputName = self._netInputName
            for j in range(len(PostModel_list[0].output_info)):
                netOutputName.append(PostModel_list[0].output_info[j].name)
            SGSSubGraphArray = self._createSubGraph(SGSTensorArray,SGSOperatorArray,subgraphname,netInputName,netOutputName,mapTensorToSubgraph)
            buf = self._createModel(SGSSubGraphArray, SGSBufferArray)
            return buf

        #lstm
        elif mode == 'concat':
            #create buffer
            tensorArray, mapBufferToTensor = self.getAllTensor(Model,mode,PostModel_list)
            SGSBufferArray_total = self._createBuffer(tensorArray)

            #create lstm subgraph
            for i in range(len(PostModel_list)):
                netInputName = []
                netOutputName = []
                for j in range(len(PostModel_list[i].input_info)):
                    netInputName.append(PostModel_list[i].input_info[j].name)
                for j in range(len(PostModel_list[i].output_info)):
                    netOutputName.append(PostModel_list[i].output_info[j].name)

                transform = SGSModel_transform_tflite.TransformToSchema(PostModel_list[i])
                PostModel_list[i],maceOpArray = transform.run()
                tensorArray, mapBufferToTensor1 = self.getAllTensor(PostModel_list[i])
                SGSTensorArray,mapTensorToSubgraph = self._createTensor(tensorArray, mapBufferToTensor)
                SGSOperatorArray = self._createOperator(PostModel_list[i],maceOpArray,mapTensorToSubgraph)
                subgraphname = PostModel_list[i].arg[0].str
                SGSSubGraphArray = self._createSubGraph(SGSTensorArray,SGSOperatorArray,subgraphname,netInputName,netOutputName,mapTensorToSubgraph)

                SubGraphArray_total = np.append(SubGraphArray_total,SGSSubGraphArray)
            #create model subgraph
            transform = SGSModel_transform_tflite.TransformToSchema(Model)
            Model,maceOpArray = transform.run()
            tensorArray, mapBufferToTensor1 = self.getAllTensor(Model)
            SGSTensorArray,mapTensorToSubgraph = self._createTensor(tensorArray,mapBufferToTensor)
            SGSOperatorArray = self._createOperator(Model,maceOpArray,mapTensorToSubgraph)
            subgraphname = Model.arg[0].str
            SGSSubGraphArray = self._createSubGraph(SGSTensorArray,SGSOperatorArray,subgraphname,self._netInputName,self._netOutputName,mapTensorToSubgraph)
            # append all subgraph
            SubGraphArray_total = np.append(SubGraphArray_total,SGSSubGraphArray)
            # create model
            buf = self._createModel(SubGraphArray_total, SGSBufferArray_total)
            return buf

        else:
            if self._platform == 'tflite':
                #tflite TransformToSchema
                transform = SGSModel_transform_tflite.TransformToSchema(Model, self._convert_kb)
            elif self._platform == 'onnx':
                #onnx TransformToSchema
                if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
                    transform = SGSModel_transform_onnx.TransformToSchema(self._net,self._netInputName,self._netInputNameShapeMap,self._netOutputName,
                                                                          self._input_name_format_map, self._input_pack_model, self._output_pack_model)
                else:
                    transform = SGSModel_transform_onnx_S.TransformToSchema(self._net,self._netInputName,self._netInputNameShapeMap,
                                                                            self._netOutputName,self._input_name_format_map, self._input_pack_model, self._output_pack_model)
            elif self._platform == 'caffe':
                #caffe TransformToSchema
                if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
                    transform = SGSModel_transform.TransformToSchema(self._net,self._netInputName,self._netInputNameShapeMap,self._netOutputName,
                                                                    self._input_pack_model, self._output_pack_model)
                else:
                    transform = SGSModel_transform_caffe_S.TransformToSchema(self._net,self._netInputName,self._netInputNameShapeMap,
                                                                              self._netOutputName, self._input_pack_model, self._output_pack_model)
                pass
            Model,maceOpArray = transform.run()
            tensorArray, mapBufferToTensor = self.getAllTensor(self._net)
            SGSBufferArray = self._createBuffer(tensorArray)
            SGSTensorArray,mapTensorToSubgraph = self._createTensor(tensorArray, mapBufferToTensor)
            SGSOperatorArray = self._createOperator(Model,maceOpArray,mapTensorToSubgraph)
            if getIPUVersion() == 'I6E' or getIPUVersion() == 'M6':
              if self._platform != 'tflite':
                CheckSGSModel = SGSModel_check_tensor_shape_from_Mace.CheckSGSModel(self._net,self._netTensorNameShapeMap,self._netTensorNameElementMap)
                CheckResult = CheckSGSModel.run()
            #subgraphname = Model.arg[0].str
            subgraphname = 'SGSModel'
            SGSSubGraphArray = self._createSubGraph(SGSTensorArray,SGSOperatorArray,subgraphname,self._netInputName,self._netOutputName,mapTensorToSubgraph)
            buf = self._createModel(SGSSubGraphArray, SGSBufferArray)
            return buf

    def tflite_getType(self, CLASS, code):
        for name, value in CLASS.__dict__.items():
            if value == code:
                return name
        return None

    def tflite_getEnum(self, CLASS, str):
        for name, value in CLASS.__dict__.items():
            if name == str:
                return value
        return None

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
        elif 'IPU_TOOL' in os.environ:
          libSGS = 'libs/x86_64/libSGSCusOP.so'
          Project_path = os.environ['IPU_TOOL']
        else:
          raise OSError('\033[31mRun `source cfg_env.sh` in Tool dir.\033[0m')
        libSGS_path = os.path.join(Project_path, libSGS)
        lib = so(libSGS_path)
        #start
        lib.startCreatFlexBuffer.restype = None
        lib.startCreatFlexBuffer()
        if name.lower().find('qirquantize') == 0:
            arg = op.arg
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == 'input_type':
                input_type = arg[i].str + '\0'
              elif name == 'num_bits':
                num_bits = arg[i].i
              elif name == 'offset_type':
                offset_type = arg[i].str + '\0'
              elif name == 'per_channel':
                per_channel = arg[i].i
              elif name == 'position_type':
                position_type = arg[i].str + '\0'
              elif name == 'scale_position_type':
                scale_position_type = arg[i].str + '\0'
              elif name == 'scale_type':
                scale_type = arg[i].str + '\0'

            #set data
            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            lib.insertKeyString(b'input_type\0',str.encode(input_type))

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'num_bits\0',num_bits)

            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            lib.insertKeyString(b'offset_type\0',str.encode(offset_type))

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'per_channel\0',per_channel)

            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            lib.insertKeyString(b'position_type\0',str.encode(position_type))

            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            lib.insertKeyString(b'scale_position_type\0',str.encode(scale_position_type))

            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            lib.insertKeyString(b'scale_type\0',str.encode(scale_type))

        if name.lower().find('elu') == 0:
            arg = op.arg
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == 'alpha':
                    alpha = arg[i].f
            # set data
            lib.insertFloatString.argtypes = [c_char_p, c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'alpha\0', alpha)

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
        elif name.lower().find('tflite_detection_nms') == 0:
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
        elif name.lower().find('sgs_lstm') == 0:
            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            c0_name = 'sub' + str(self._lstm_num) + '_add_3#output\0'
            lib.insertKeyString(b'C0\0', str.encode(c0_name))

            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            sub_lstm_name = 'SGS_LSTM_sub' + str(self._lstm_num) + '\0'
            self._lstm_num += 1
            lib.insertKeyString(b'SubNet\0', str.encode(sub_lstm_name))

            arg = op.arg
            lstm_type = 0
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == "lstm_type":
                  lstm_type = arg[i].i
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'type\0',lstm_type)
        elif name.lower().find('sgs_gru') == 0:
            #lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            #lib.insertFloatString.restype = None
            #c0_name = 'sub' + str(self._lstm_num) + '_add_3#output\0'
            #lib.insertKeyString(b'C0\0', str.encode(c0_name))

            lib.insertFloatString.argtypes = [c_char_p,c_char_p]
            lib.insertFloatString.restype = None
            sub_gru_name = 'SGS_GRU_sub' + str(self._gru_num) + '\0'
            self._gru_num += 1
            lib.insertKeyString(b'SubNet\0', str.encode(sub_gru_name))

            arg = op.arg
            gru_type = 0
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == "gru_type":
                  gru_type = arg[i].i
            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'type\0',gru_type)

        elif name.lower().find('groupconv') == 0:
            arg = op.arg
            strideH,strideW = 0,0
            dilationH,dilationW = 1,1
            fusedActivationFunction = 0
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_padding_values_str:
                    paddingL,paddingT,paddingR,paddingB = arg[i].ints
                elif name == MaceKeyword.mace_strides_str:
                    strideH,strideW = arg[i].ints
                elif name == MaceKeyword.mace_padding_str:
                    padding = arg[i].i
                elif name == MaceKeyword.mace_dilations_str:
                    dilationH,dilationW = arg[i].ints
                elif name == 'group':
                    group = arg[i].i
                elif name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
                    fusedActivationFunction = arg[i].i

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'Padding\0',padding)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'fused_activation_function\0',fusedActivationFunction)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'group\0',group)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'dilation_h\0',dilationH)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'dilation_w\0',dilationW)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_w\0',strideW)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_h\0',strideH)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_left\0',paddingL)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_right\0',paddingR)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_top\0',paddingT)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_bottom\0',paddingB)


        elif name.lower().find('maxpool3d') == 0:
            arg = op.arg
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_padding_values_str:
                    paddingL,paddingT,paddingI,paddingR,paddingB,paddingO = arg[i].ints
                elif name == MaceKeyword.mace_strides_str:
                    strideD,strideH,strideW = arg[i].ints
                elif name == MaceKeyword.mace_kernel_str:
                    kernelsD,kernelsH,kernelsW = arg[i].ints

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_d\0',strideD)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_w\0',strideW)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_h\0',strideH)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'filter_depth\0',kernelsD)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'filter_width\0',kernelsW)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'filter_height\0',kernelsH)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_depth\0',paddingI)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_depth_offset\0',paddingO-paddingI)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_width\0',paddingL)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_width_offset\0',paddingR-paddingL)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_height\0',paddingB)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_height_offset\0',paddingT-paddingB)

        elif name.lower().find('avepool3d') == 0:
            arg = op.arg
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_padding_values_str:
                    paddingL,paddingT,paddingI,paddingR,paddingB,paddingO = arg[i].ints
                elif name == MaceKeyword.mace_strides_str:
                    strideD,strideH,strideW = arg[i].ints
                elif name == MaceKeyword.mace_kernel_str:
                    kernelsD,kernelsH,kernelsW = arg[i].ints

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_d\0',strideD)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_w\0',strideW)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'stride_h\0',strideH)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'filter_depth\0',kernelsD)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'filter_width\0',kernelsW)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'filter_height\0',kernelsH)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_depth\0',paddingI)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_depth_offset\0',paddingO-paddingI)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_width\0',paddingL)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_width_offset\0',paddingR-paddingL)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_height\0',paddingB)

            lib.insertIntString.argtypes = [c_char_p,c_int]
            lib.insertIntString.restype = None
            lib.insertIntString(b'padding_height_offset\0',paddingT-paddingB)
        elif name.lower().find('sgs_roipooling') == 0:
            #set data
            arg = op.arg
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == 'spatial_scale':
                spatial_scale = arg[i].f
            lib.insertFloatString.argtypes = [c_char_p,c_float]
            lib.insertFloatString.restype = None
            lib.insertFloatString(b'spatial_scale\0', spatial_scale)
        elif name.lower().find('customized_scatternd') == 0:
            arg = op.arg
            for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == 'reduction':
                    reduction = arg[i].i
            lib.insertIntString.argtypes = [c_char_p, c_int]
            lib.insertFloatString.restype = None
            lib.insertIntString(b'reduction\0', reduction)
        elif name == 'Custom':
            arg = op.arg
            customer_options = None
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == 'custom_options':
                customer_options = arg[i].s
            mace_check(customer_options is not None, "this Custom:%s layer will be support comming soon"%name)
            return customer_options

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
        #create option
        tflite.AbsOptions.AbsOptionsStart(self._builder)
        builtinOp = tflite.AbsOptions.AbsOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().ABS)
        if 'ABS' not in self._operatorCodeArray:
          self._operatorCodeArray['ABS'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_ADD(self, op):
        fusedActivationFunction = 0
        #merge follow Activation op
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().AddOptions
        potScaleInt16 = 1
        arg = op.arg
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
                fusedActivationFunction = arg[i].i
            elif name == 'PotScaleInt16':
                potScaleInt16 = arg[i].i
        #creat option
        tflite.AddOptions.AddOptionsStart(self._builder)
        tflite.AddOptions.AddOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        tflite.AddOptions.AddOptionsAddPotScaleInt16(self._builder,potScaleInt16)
        builtinOp = tflite.AddOptions.AddOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().ADD)
        if 'ADD' not in self._operatorCodeArray:
          self._operatorCodeArray['ADD'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_ADD_N(self, op):
        #merge follow Activation op
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().AddNOptions
        #create option
        tflite.AddNOptions.AddNOptionsStart(self._builder)
        builtinOp = tflite.AddNOptions.AddNOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().ADD_N)
        if 'ADD_N' not in self._operatorCodeArray:
          self._operatorCodeArray['ADD_N'] = operatorCode

    def builtin_CONCATENATION(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions
        fusedActivationFunction = 0
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_axis_str:
            axis = arg[i].i
          elif name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
            fusedActivationFunction = arg[i].i
        #create option
        tflite.ConcatenationOptions.ConcatenationOptionsStart(self._builder)
        tflite.ConcatenationOptions.ConcatenationOptionsAddAxis(self._builder,axis)
        tflite.ConcatenationOptions.ConcatenationOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        builtinOp = tflite.ConcatenationOptions.ConcatenationOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().CONCATENATION)
        if 'CONCATENATION' not in self._operatorCodeArray:
          self._operatorCodeArray['CONCATENATION'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_CONV_2D(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().Conv2DOptions
        arg = op.arg
        strideH,strideW = 0,0
        dilationH,dilationW = 1,1
        fusedActivationFunction = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            #padding value should be divided by 2
            paddingL,paddingR,paddingT,paddingB = arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideH,strideW = arg[i].ints
          elif name == MaceKeyword.mace_dilations_str:
            dilationH,dilationW = arg[i].ints
          elif name == MaceKeyword.mace_padding_str:
            padding = arg[i].i
          elif name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
            fusedActivationFunction = arg[i].i
        #create option
        tflite.Conv2DOptions.Conv2DOptionsStart(self._builder)
        tflite.Conv2DOptions.Conv2DOptionsAddPadding(self._builder,padding)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.Conv2DOptions.Conv2DOptionsAddStrideW(self._builder,strideW)
        tflite.Conv2DOptions.Conv2DOptionsAddStrideH(self._builder,strideH)
        tflite.Conv2DOptions.Conv2DOptionsAddDilationWFactor(self._builder,dilationW)
        tflite.Conv2DOptions.Conv2DOptionsAddDilationHFactor(self._builder,dilationH)
        tflite.Conv2DOptions.Conv2DOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        builtinOp = tflite.Conv2DOptions.Conv2DOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().CONV_2D)
        if 'CONV_2D' not in self._operatorCodeArray:
          self._operatorCodeArray['CONV_2D'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_DEPTHWISE_CONV_2D(self, op):
        arg = op.arg
        strideH,strideW = 0,0
        dilationH,dilationW = 1,1
        fusedActivationFunction = 0
        DepthMultiplier = 0
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
          elif name == MaceKeyword.mace_padding_str:
            padding = arg[i].i
          elif name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
            fusedActivationFunction = arg[i].i

        #create option
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsStart(self._builder)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPadding(self._builder,padding)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideW(self._builder,strideW)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideH(self._builder,strideH)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationWFactor(self._builder,dilationW)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationHFactor(self._builder,dilationH)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDepthMultiplier(self._builder,DepthMultiplier)
        builtinOp = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().DEPTHWISE_CONV_2D)
        if 'DEPTHWISE_CONV_2D' not in self._operatorCodeArray:
          self._operatorCodeArray['DEPTHWISE_CONV_2D'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_CONV_3D(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().Conv3DOptions
        fusedActivationFunction = 0
        arg = op.arg
        strideD,strideH,strideW = 0,0,0
        dilationD,dilationH,dilationW = 1,1,1
        paddingL,paddingR,paddingT,paddingB,paddingI,paddingO = 0,0,0,0,0,0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            paddingI,paddingL,paddingT,paddingO,paddingR,paddingB = arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideD,strideH,strideW = arg[i].ints
          elif name == MaceKeyword.mace_dilations_str:
            dilationD,dilationH,dilationW = arg[i].ints
          elif name == MaceKeyword.mace_padding_str:
            padding = arg[i].i
          elif name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
            fusedActivationFunction = arg[i].i
        #create option
        tflite.Conv3DOptions.Conv3DOptionsStart(self._builder)
        tflite.Conv3DOptions.Conv3DOptionsAddPadding(self._builder,padding)
        tflite.Conv3DOptions.Conv3DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.Conv3DOptions.Conv3DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.Conv3DOptions.Conv3DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.Conv3DOptions.Conv3DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.Conv3DOptions.Conv3DOptionsAddPaddingInside(self._builder,paddingI)
        tflite.Conv3DOptions.Conv3DOptionsAddPaddingOutside(self._builder,paddingO)
        tflite.Conv3DOptions.Conv3DOptionsAddStrideD(self._builder,strideD)
        tflite.Conv3DOptions.Conv3DOptionsAddStrideW(self._builder,strideW)
        tflite.Conv3DOptions.Conv3DOptionsAddStrideH(self._builder,strideH)
        tflite.Conv3DOptions.Conv3DOptionsAddDilationDFactor(self._builder,dilationD)
        tflite.Conv3DOptions.Conv3DOptionsAddDilationWFactor(self._builder,dilationW)
        tflite.Conv3DOptions.Conv3DOptionsAddDilationHFactor(self._builder,dilationH)
        tflite.Conv3DOptions.Conv3DOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        builtinOp = tflite.Conv3DOptions.Conv3DOptionsEnd(self._builder)

        #creat Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().CONV_3D)
        if 'CONV_3D' not in self._operatorCodeArray:
          self._operatorCodeArray['CONV_3D'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_DEQUANTIZE(self, arg):


        pass
    def builtin_EMBEDDING_LOOKUP(self, arg):


        pass
    def builtin_FLOOR(self, arg):
        pass
    def builtin_ACTIVATION(self,op):
        pass
    def builtin_FULLY_CONNECTED(self, op):
        arg = op.arg
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().FullyConnectedOptions
        WeightsFormat = 0
        fusedActivationFunction = 0
        keepNumDims = 0
        asymmetricQuantizeInputs = 0
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
                fusedActivationFunction = arg[i].i
            elif name == 'SHUFFLED4x16INT8':
                WeightsFormat = arg[i].i
            elif name == 'KeepNumDims':
                keepNumDims = arg[i].i
            elif name == 'AsymmetricQuantizeInputs':
                asymmetricQuantizeInputs = arg[i].i
        #create option
        tflite.FullyConnectedOptions.FullyConnectedOptionsStart(self._builder)
        tflite.FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        tflite.FullyConnectedOptions.FullyConnectedOptionsAddWeightsFormat(self._builder,WeightsFormat)
        tflite.FullyConnectedOptions.FullyConnectedOptionsAddKeepNumDims(self._builder,keepNumDims)
        tflite.FullyConnectedOptions.AddAsymmetricQuantizeInputs(self._builder,asymmetricQuantizeInputs)
        builtinOp = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().FULLY_CONNECTED)
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
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
        if 'LOGISTIC' not in self._operatorCodeArray:
          self._operatorCodeArray['LOGISTIC'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_LSH_PROJECTION(self, arg):


        pass
    def builtin_LSTM(self, arg):
        pass
    def builtin_GRU(self, arg):
        pass
    def builtin_AVERAGE_POOL_2D(self,op):
        arg = op.arg
        strideW,strideH,filter_width,filter_height,paddingW,paddingH= 0,0,0,0,0,0
        fusedActivationFunction = 0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().Pool2DOptions
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            paddingL,paddingR,paddingT,paddingB = arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideH,strideW = arg[i].ints
          elif name == MaceKeyword.mace_kernel_str:
            filter_height,filter_width= arg[i].ints
          elif name == MaceKeyword.mace_padding_str:
            padding = arg[i].i
          elif name == MaceKeyword.mace_pooling_type_str:
            pooling_type = arg[i].i
          elif name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
            fusedActivationFunction = arg[i].i
        tflite.Pool2DOptions.Pool2DOptionsStart(self._builder)
        tflite.Pool2DOptions.Pool2DOptionsAddPadding(self._builder,padding)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideW(self._builder,strideW)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideH(self._builder,strideH)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(self._builder,filter_width)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(self._builder,filter_height)
        tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        builtinOp = tflite.Pool2DOptions.Pool2DOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().AVERAGE_POOL_2D)
        if 'AVERAGE_POOL_2D' not in self._operatorCodeArray:
          self._operatorCodeArray['AVERAGE_POOL_2D'] = operatorCode

        return builtinOpTypes,builtinOp

    def builtin_MAX_POOL_2D(self, op):
        arg = op.arg
        strideW,strideH,filter_width,filter_height= 0,0,0,0
        fusedActivationFunction = 0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().Pool2DOptions
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            paddingL,paddingR,paddingT,paddingB = arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideH,strideW = arg[i].ints
          elif name == MaceKeyword.mace_kernel_str:
            filter_height,filter_width = arg[i].ints
          elif name == MaceKeyword.mace_padding_str:
            padding = arg[i].i
          elif name == MaceKeyword.mace_pooling_type_str:
            pooling_type = arg[i].i
          elif name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
            fusedActivationFunction = arg[i].i
        tflite.Pool2DOptions.Pool2DOptionsStart(self._builder)
        tflite.Pool2DOptions.Pool2DOptionsAddPadding(self._builder,padding)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingLeft(self._builder,paddingL)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingRight(self._builder,paddingR)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingTop(self._builder,paddingT)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingBottom(self._builder,paddingB)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideW(self._builder,strideW)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideH(self._builder,strideH)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(self._builder,filter_width)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(self._builder,filter_height)
        tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        builtinOp = tflite.Pool2DOptions.Pool2DOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().MAX_POOL_2D)
        if 'MAX_POOL_2D' not in self._operatorCodeArray:
          self._operatorCodeArray['MAX_POOL_2D'] = operatorCode
        return builtinOpTypes,builtinOp


    def builtin_MUL(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().MulOptions
        #creat option
        fusedActivationFunction = 0
        arg = op.arg
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
                fusedActivationFunction = arg[i].i
        tflite.MulOptions.MulOptionsStart(self._builder)
        tflite.MulOptions.MulOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        builtinOp = tflite.MulOptions.MulOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().MUL)
        if 'MUL' not in self._operatorCodeArray:
          self._operatorCodeArray['MUL'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RELU(self, op):
        builtinOp = None
        builtinOpTypes = None
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().RELU)
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

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().LEAKY_RELU)
        if 'LEAKY_RELU' not in self._operatorCodeArray:
          self._operatorCodeArray['LEAKY_RELU'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RELU6(self, op):
        builtinOp = None
        builtinOpTypes = None
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().RELU6)
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
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
        if 'RESHAPE' not in self._operatorCodeArray:
          self._operatorCodeArray['RESHAPE'] = operatorCode
        return builtinOpTypes,builtinOp


    def builtin_RESIZE_BILINEAR(self, op):
        arg = op.arg
        align_corners = 0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ResizeBilinearOptions
        halfPixelCenters = 0

        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_align_corners_str:
            align_corners = arg[i].i
          elif name == MaceKeyword.mace_HalfPixelCenters_str:
            halfPixelCenters = arg[i].i
        #creat option
        tflite.ResizeBilinearOptions.ResizeBilinearOptionsStart(self._builder)
        tflite.ResizeBilinearOptions.ResizeBilinearOptionsAddHalfPixelCenters(self._builder,halfPixelCenters)
        tflite.ResizeBilinearOptions.ResizeBilinearOptionsAddAlignCorners(self._builder,align_corners)
        builtinOp = tflite.ResizeBilinearOptions.ResizeBilinearOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().RESIZE_BILINEAR)
        if 'RESIZE_BILINEAR' not in self._operatorCodeArray:
          self._operatorCodeArray['RESIZE_BILINEAR'] = operatorCode
        return builtinOpTypes,builtinOp


    def builtin_RESIZE_NEAREST_NEIGHBOR(self, op):
        arg = op.arg
        align_corners = 0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ResizeNearestNeighborOptions
        halfPixelCenters = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'align_corners':
            #padding value should be divided by 2
            align_corners = arg[i].i
        #create option
        tflite.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsStart(self._builder)
        tflite.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsAddHalfPixelCenters(self._builder,halfPixelCenters)
        tflite.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsAddAlignCorners(self._builder,align_corners)
        builtinOp = tflite.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().RESIZE_NEAREST_NEIGHBOR)
        if 'RESIZE_NEAREST_NEIGHBOR' not in self._operatorCodeArray:
          self._operatorCodeArray['RESIZE_NEAREST_NEIGHBOR'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RNN(self, op):
        pass

    def builtin_SOFTMAX(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SoftmaxOptions
        beta = 0.0
        #create option
        tflite.SoftmaxOptions.SoftmaxOptionsStart(self._builder)
        tflite.SoftmaxOptions.SoftmaxOptionsAddBeta(self._builder,beta)
        builtinOp = tflite.SoftmaxOptions.SoftmaxOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SOFTMAX)
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
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().TANH)
        if 'TANH' not in self._operatorCodeArray:
          self._operatorCodeArray['TANH'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_EMBEDDING_LOOKUP_SPARSE(self, op):


        pass
    def builtin_PAD(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().PadOptions
        tflite.PadOptions.PadOptionsStart(self._builder)
        builtinOp = tflite.PadOptions.PadOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().PAD)
        if 'PAD' not in self._operatorCodeArray:
          self._operatorCodeArray['PAD'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_MIRROR_PAD(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().MirrorPadOptions
        tflite.MirrorPadOptions.MirrorPadOptionsStart(self._builder)
        tflite.MirrorPadOptions.MirrorPadOptionsAddMode(self._builder, tflite.MirrorPadMode.MirrorPadMode.REFLECT)
        builtinOp = tflite.MirrorPadOptions.MirrorPadOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().MIRROR_PAD)
        if 'MIRROR_PAD' not in self._operatorCodeArray:
          self._operatorCodeArray['MIRROR_PAD'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_UNIDIRECTIONAL_SEQUENCE_RNN(self, op):


        pass
    def builtin_GATHER(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().GatherOptions
        batchDims = 0
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'batchDims':
            batchDims = arg[i].i
          elif name == MaceKeyword.mace_axis_str:
            axis = arg[i].i
        #creat option
        tflite.GatherOptions.GatherOptionsStart(self._builder)
        tflite.GatherOptions.GatherOptionsAddAxis(self._builder,axis)
        tflite.GatherOptions.GatherOptionsAddBatchDims(self._builder,batchDims)
        builtinOp = tflite.GatherOptions.GatherOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().GATHER)
        if 'GATHER' not in self._operatorCodeArray:
          self._operatorCodeArray['GATHER'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_BATCH_TO_SPACE_ND(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().BatchToSpaceNDOptions
        tflite.BatchToSpaceNDOptions.BatchToSpaceNDOptionsStart(self._builder)
        builtinOp = tflite.BatchToSpaceNDOptions.BatchToSpaceNDOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().BATCH_TO_SPACE_ND)
        if 'BATCH_TO_SPACE_ND' not in self._operatorCodeArray:
          self._operatorCodeArray['BATCH_TO_SPACE_ND'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_SPACE_TO_BATCH_ND(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SpaceToBatchNDOptions
        tflite.SpaceToBatchNDOptions.SpaceToBatchNDOptionsStart(self._builder)
        builtinOp = tflite.SpaceToBatchNDOptions.SpaceToBatchNDOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SPACE_TO_BATCH_ND)
        if 'SPACE_TO_BATCH_ND' not in self._operatorCodeArray:
          self._operatorCodeArray['SPACE_TO_BATCH_ND'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_TRANSPOSE(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().TransposeOptions
        tflite.TransposeOptions.TransposeOptionsStart(self._builder)
        builtinOp = tflite.TransposeOptions.TransposeOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().TRANSPOSE)
        if 'TRANSPOSE' not in self._operatorCodeArray:
          self._operatorCodeArray['TRANSPOSE'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_MEAN(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ReducerOptions
        #creat option
        arg = op.arg
        keep_dim = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_keepdims_str:
            keep_dim = arg[i].i

        tflite.ReducerOptions.ReducerOptionsStart(self._builder)
        tflite.ReducerOptions.ReducerOptionsAddKeepDims(self._builder,keep_dim)
        builtinOp = tflite.ReducerOptions.ReducerOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().MEAN)
        if 'MEAN' not in self._operatorCodeArray:
          self._operatorCodeArray['MEAN'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_SUB(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SubOptions
        potScaleInt16 = 1
        fusedActivationFunction = 0
        arg = op.arg
        for i in six.moves.range(len(arg)):
            name = arg[i].name
            if name == 'RELU6' or name == 'RELU' or name == 'RELU_N1_TO_1' or name == 'TANH' or name == 'SIGN_BIT':
                fusedActivationFunction = arg[i].i
            elif name == 'PotScaleInt16':
                potScaleInt16 = arg[i].i
        tflite.SubOptions.SubOptionsStart(self._builder)
        tflite.SubOptions.SubOptionsAddFusedActivationFunction(self._builder,fusedActivationFunction)
        tflite.SubOptions.SubOptionsAddPotScaleInt16(self._builder,potScaleInt16)
        builtinOp = tflite.SubOptions.SubOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SUB)
        if 'SUB' not in self._operatorCodeArray:
          self._operatorCodeArray['SUB'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_DIV(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().DivOptions
        #creat option
        tflite.DivOptions.DivOptionsStart(self._builder)
        tflite.DivOptions.DivOptionsAddFusedActivationFunction(self._builder,tflite.ActivationFunctionType.ActivationFunctionType().NONE)
        builtinOp = tflite.DivOptions.DivOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().DIV)
        if 'DIV' not in self._operatorCodeArray:
          self._operatorCodeArray['DIV'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_SQUEEZE(self, op):


        pass
    def builtin_UNIDIRECTIONAL_SEQUENCE_LSTM(self, op):


        pass

    def builtin_BIDIRECTIONAL_SEQUENCE_RNN(self, op):


        pass
    def builtin_EXP(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ExpOptions
        tflite.ExpOptions.ExpOptionsStart(self._builder)
        builtinOp = tflite.ExpOptions.ExpOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().EXP)
        if 'EXP' not in self._operatorCodeArray:
          self._operatorCodeArray['EXP'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_TOPK_V2(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().TopKV2Options
        tflite.TopKV2Options.TopKV2OptionsStart(self._builder)
        builtinOp = tflite.TopKV2Options.TopKV2OptionsEnd(self._builder)

        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().TOPK_V2)
        if 'TOPK_V2' not in self._operatorCodeArray:
            self._operatorCodeArray['TOPK_V2'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_SPLIT(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SplitOptions
        # create option
        arg = op.arg
        numSplits = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_num_split_str:
            numSplits = arg[i].i
        tflite.SplitOptions.SplitOptionsStart(self._builder)
        tflite.SplitOptions.SplitOptionsAddNumSplits(self._builder, numSplits)
        builtinOp = tflite.SplitOptions.SplitOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SPLIT)
        if 'SPLIT' not in self._operatorCodeArray:
          self._operatorCodeArray['SPLIT'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_SPLIT_V(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SplitVOptions
        # create option
        arg = op.arg
        numSplits = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_num_split_str:
            numSplits = arg[i].i
        tflite.SplitVOptions.SplitVOptionsStart(self._builder)
        tflite.SplitVOptions.SplitVOptionsAddNumSplits(self._builder, numSplits)
        builtinOp = tflite.SplitVOptions.SplitVOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SPLIT_V)
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
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().CastOptions
        tflite.CastOptions.CastOptionsStart(self._builder)
        builtinOp = tflite.CastOptions.CastOptionsEnd(self._builder)
        #creat Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().CAST)
        if 'CAST' not in self._operatorCodeArray:
          self._operatorCodeArray['CAST'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_PRELU(self, op):
        builtinOpTypes = None
        builtinOp = None
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().PRELU)
        if 'PRELU' not in self._operatorCodeArray:
          self._operatorCodeArray['PRELU'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_MAXIMUM(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().MaximumMinimumOptions
        tflite.MaximumMinimumOptions.MaximumMinimumOptionsStart(self._builder)
        builtinOp = tflite.MaximumMinimumOptions.MaximumMinimumOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().MAXIMUM)
        if 'MAXIMUM' not in self._operatorCodeArray:
          self._operatorCodeArray['MAXIMUM'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_COS(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().CosOptions
        tflite.CosOptions.CosOptionsStart(self._builder)
        builtinOp = tflite.CosOptions.CosOptionsEnd(self._builder)

        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().COS)
        if 'COS' not in self._operatorCodeArray:
            self._operatorCodeArray['COS'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_SIN(self, op):
        builtinOpTypes = None
        builtinOp = None
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SIN)
        if 'SIN' not in self._operatorCodeArray:
          self._operatorCodeArray['SIN'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_ARG_MAX(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ArgMaxOptions
        arg = op.arg
        OutputType = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_output_type_str:
            OutputType = arg[i].i
        tflite.ArgMaxOptions.ArgMaxOptionsStart(self._builder)
        tflite.ArgMaxOptions.ArgMaxOptionsAddOutputType(self._builder, OutputType)
        builtinOp = tflite.ArgMaxOptions.ArgMaxOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().ARG_MAX)
        if 'ARG_MAX' not in self._operatorCodeArray:
          self._operatorCodeArray['ARG_MAX'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_MINIMUM(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().MaximumMinimumOptions
        tflite.MaximumMinimumOptions.MaximumMinimumOptionsStart(self._builder)
        builtinOp = tflite.MaximumMinimumOptions.MaximumMinimumOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().MINIMUM)
        if 'MINIMUM' not in self._operatorCodeArray:
          self._operatorCodeArray['MINIMUM'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_LESS(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().LessOptions
        tflite.LessOptions.LessOptionsStart(self._builder)
        builtinOp = tflite.LessOptions.LessOptionsEnd(self._builder)

        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().LESS)
        if 'LESS' not in self._operatorCodeArray:
            self._operatorCodeArray['LESS'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_NEG(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().NegOptions
        tflite.NegOptions.NegOptionsStart(self._builder)
        builtinOp = tflite.NegOptions.NegOptionsEnd(self._builder)

        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().NEG)
        if 'NEG' not in self._operatorCodeArray:
            self._operatorCodeArray['NEG'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_PADV2(self, op):


        pass
    def builtin_GREATER(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().GreaterOptions
        tflite.GreaterOptions.GreaterOptionsStart(self._builder)
        builtinOp = tflite.GreaterOptions.GreaterOptionsEnd(self._builder)

        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().GREATER)
        if 'GREATER' not in self._operatorCodeArray:
            self._operatorCodeArray['GREATER'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_ELU(self, op):
        builtinOpTypes = None
        builtinOp = None
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().ELU)
        if 'ELU' not in self._operatorCodeArray:
            self._operatorCodeArray['ELU'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_GREATER_EQUAL(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().GreaterEqualOptions
        tflite.GreaterEqualOptions.GreaterEqualOptionsStart(self._builder)
        builtinOp = tflite.GreaterEqualOptions.GreaterEqualOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().GREATER_EQUAL)
        if 'GREATER_EQUAL' not in self._operatorCodeArray:
          self._operatorCodeArray['GREATER_EQUAL'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_LESS_EQUAL(self, op):


        pass
    def builtin_SELECT(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SelectOptions
        tflite.SelectOptions.SelectOptionsStart(self._builder)
        builtinOp = tflite.SelectOptions.SelectOptionsEnd(self._builder)

        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SELECT)
        if 'SELECT' not in self._operatorCodeArray:
            self._operatorCodeArray['SELECT'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_SLICE(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().SliceOptions
        tflite.SliceOptions.SliceOptionsStart(self._builder)
        builtinOp = tflite.SliceOptions.SliceOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SLICE)
        if 'SLICE' not in self._operatorCodeArray:
          self._operatorCodeArray['SLICE'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_STRIDEDSLICE(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().StridedSliceOptions
        beginMask = 0
        endMask = 0
        ellipsisMask = 0
        newAxisMask = 0
        shrinkAxisMask = 0
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_begin_mask_str:
            beginMask = arg[i].i
          if name == MaceKeyword.mace_end_mask_str:
            endMask = arg[i].i
          if name == MaceKeyword.mace_ellipsis_mask_str:
            ellipsisMask = arg[i].i
          if name == MaceKeyword.mace_new_axis_mask_str:
            newAxisMask = arg[i].i
          if name == MaceKeyword.mace_shrink_axis_mask_str:
            shrinkAxisMask = arg[i].i
        #creat option
        tflite.StridedSliceOptions.StridedSliceOptionsStart(self._builder)
        tflite.StridedSliceOptions.StridedSliceOptionsAddBeginMask(self._builder, beginMask)
        tflite.StridedSliceOptions.StridedSliceOptionsAddEndMask(self._builder, endMask)
        tflite.StridedSliceOptions.StridedSliceOptionsAddEllipsisMask(self._builder, ellipsisMask)
        tflite.StridedSliceOptions.StridedSliceOptionsAddNewAxisMask(self._builder, newAxisMask)
        tflite.StridedSliceOptions.StridedSliceOptionsAddShrinkAxisMask(self._builder, shrinkAxisMask)
        builtinOp = tflite.StridedSliceOptions.StridedSliceOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().STRIDED_SLICE)
        if 'STRIDED_SLICE' not in self._operatorCodeArray:
          self._operatorCodeArray['STRIDED_SLICE'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_BATCHMATMUL(self, op):
        asymmetricQuantizeInputs = 0
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().BatchMatMulOptions
        changeA = False
        changeB = False
        #creat option
        tflite.BatchMatMulOptions.BatchMatMulOptionsStart(self._builder)
        tflite.BatchMatMulOptions.BatchMatMulOptionsAddAdjX(self._builder, changeA)
        tflite.BatchMatMulOptions.BatchMatMulOptionsAddAdjY(self._builder, changeB)
        tflite.BatchMatMulOptions.BatchMatMulOptionsAddAsymmetricQuantizeInputs(self._builder, asymmetricQuantizeInputs)
        builtinOp = tflite.BatchMatMulOptions.BatchMatMulOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().BATCH_MATMUL)
        if 'BATCH_MATMUL' not in self._operatorCodeArray:
          self._operatorCodeArray['BATCH_MATMUL'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_TRANSPOSE_CONV(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().TransposeConvOptions
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_str:
            padding_type = arg[i].i
          if name == 'stride_w':
            stride_w = arg[i].i
          if name == 'stride_h':
            stride_h = arg[i].i
          if name == 'padding_left':
            padding_left = arg[i].i
          if name == 'padding_right':
            padding_right = arg[i].i
          if name == 'padding_top':
            padding_top = arg[i].i
          if name == 'padding_bottom':
            padding_bottom = arg[i].i
        #creat option
        tflite.TransposeConvOptions.TransposeConvOptionsStart(self._builder)
        tflite.TransposeConvOptions.TransposeConvOptionsAddPadding(self._builder, padding_type)
        tflite.TransposeConvOptions.TransposeConvOptionsAddStrideW(self._builder, stride_w)
        tflite.TransposeConvOptions.TransposeConvOptionsAddStrideH(self._builder, stride_h)
        tflite.TransposeConvOptions.TransposeConvOptionsAddPaddingLeft(self._builder, padding_left)
        tflite.TransposeConvOptions.TransposeConvOptionsAddPaddingRight(self._builder, padding_right)
        tflite.TransposeConvOptions.TransposeConvOptionsAddPaddingTop(self._builder, padding_top)
        tflite.TransposeConvOptions.TransposeConvOptionsAddPaddingBottom(self._builder, padding_bottom)
        builtinOp = tflite.TransposeConvOptions.TransposeConvOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().TRANSPOSE_CONV)
        if 'TRANSPOSE_CONV' not in self._operatorCodeArray:
          self._operatorCodeArray['TRANSPOSE_CONV'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_SPARSE_TO_DENSE(self, op):


        pass
    def builtin_TILE(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().TileOptions
        tflite.TileOptions.TileOptionsStart(self._builder)
        builtinOp = tflite.TileOptions.TileOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().TILE)
        if 'TILE' not in self._operatorCodeArray:
          self._operatorCodeArray['TILE'] = operatorCode
        return builtinOpTypes,builtinOp



    def builtin_EXPAND_DIMS(self, op):


        pass
    def builtin_EQUAL(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().EqualOptions
        tflite.EqualOptions.EqualOptionsStart(self._builder)
        builtinOp = tflite.EqualOptions.EqualOptionsEnd(self._builder)
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().EQUAL)
        if 'EQUAL' not in self._operatorCodeArray:
          self._operatorCodeArray['EQUAL'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_NOT_EQUAL(self, op):

        pass

    def builtin_LOG(self, op):
        builtinOp = None
        builtinOpTypes = None
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().LOG)
        if 'LOG' not in self._operatorCodeArray:
          self._operatorCodeArray['LOG'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_SUM(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ReducerOptions
        #creat option
        arg = op.arg
        keep_dim = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_keepdims_str:
            keep_dim = arg[i].i

        tflite.ReducerOptions.ReducerOptionsStart(self._builder)
        tflite.ReducerOptions.ReducerOptionsAddKeepDims(self._builder,keep_dim)
        builtinOp = tflite.ReducerOptions.ReducerOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SUM)
        if 'SUM' not in self._operatorCodeArray:
          self._operatorCodeArray['SUM'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_SQRT(self, op):
        builtinOp = None
        builtinOpTypes = None
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SQRT)
        if 'SQRT' not in self._operatorCodeArray:
          self._operatorCodeArray['SQRT'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_RSQRT(self, op):
        builtinOp = None
        builtinOpTypes = None
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().RSQRT)
        if 'RSQRT' not in self._operatorCodeArray:
          self._operatorCodeArray['RSQRT'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_SHAPE(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ShapeOptions
        tflite.ShapeOptions.ShapeOptionsStart(self._builder)
        builtinOp = tflite.ShapeOptions.ShapeOptionsEnd(self._builder)
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SHAPE)
        if 'SHAPE' not in self._operatorCodeArray:
          self._operatorCodeArray['SHAPE'] = operatorCode
        return builtinOpTypes,builtinOp

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
        keep_dim = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_keepdims_str:
            keep_dim = arg[i].i

        tflite.ReducerOptions.ReducerOptionsStart(self._builder)
        tflite.ReducerOptions.ReducerOptionsAddKeepDims(self._builder,keep_dim)
        builtinOp = tflite.ReducerOptions.ReducerOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().REDUCE_MAX)
        if 'REDUCE_MAX' not in self._operatorCodeArray:
          self._operatorCodeArray['REDUCE_MAX'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_REDUCE_MIN(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().ReducerOptions
        #creat option
        arg = op.arg
        keep_dim = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_keepdims_str:
            keep_dim = arg[i].i

        tflite.ReducerOptions.ReducerOptionsStart(self._builder)
        tflite.ReducerOptions.ReducerOptionsAddKeepDims(self._builder,keep_dim)
        builtinOp = tflite.ReducerOptions.ReducerOptionsEnd(self._builder)
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().REDUCE_MIN)
        if 'REDUCE_MIN' not in self._operatorCodeArray:
          self._operatorCodeArray['REDUCE_MIN'] = operatorCode
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

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().PACK)
        if 'PACK' not in self._operatorCodeArray:
          self._operatorCodeArray['PACK'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_LOGICAL_OR(self, op):


        pass
    def builtin_ONE_HOT(self, op):


        pass
    def builtin_LOGICAL_AND(self, op):
        tflite.LogicalAndOptions.LogicalAndOptionsStart(self._builder)
        builtinOp = tflite.LogicalAndOptions.LogicalAndOptionsEnd(self._builder)
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().LogicalAndOptions
        #creat Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().LOGICAL_AND)
        if 'LOGICAL_AND' not in self._operatorCodeArray:
          self._operatorCodeArray['LOGICAL_AND'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_LOGICAL_NOT(self, op):
        tflite.LogicalNotOptions.LogicalNotOptionsStart(self._builder)
        builtinOp = tflite.LogicalNotOptions.LogicalNotOptionsEnd(self._builder)
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().LogicalNotOptions
        #creat Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().LOGICAL_NOT)
        if 'LOGICAL_NOT' not in self._operatorCodeArray:
          self._operatorCodeArray['LOGICAL_NOT'] = operatorCode
        return builtinOpTypes,builtinOp

    def builtin_ROUND(self, op):
        builtinOp = None
        builtinOpTypes = None
        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().ROUND)
        if 'ROUND' not in self._operatorCodeArray:
          self._operatorCodeArray['ROUND'] = operatorCode
        return builtinOpTypes,builtinOp

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
        tflite.UnpackOptions.UnpackOptionsStart(self._builder)
        tflite.UnpackOptions.UnpackOptionsAddAxis(self._builder,axis)
        tflite.UnpackOptions.UnpackOptionsAddNum(self._builder,num)
        builtinOp = tflite.UnpackOptions.UnpackOptionsEnd(self._builder)

        #create Operator Code
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().UNPACK)
        if 'UNPACK' not in self._operatorCodeArray:
          self._operatorCodeArray['UNPACK'] = operatorCode
        return builtinOpTypes,builtinOp


    def builtin_FLOOR_DIV(self, op):
        builtinOpTypes = tflite.BuiltinOptions.BuiltinOptions().FloorDivOptions
        tflite.FloorDivOptions.FloorDivOptionsStart(self._builder)
        builtinOp = tflite.FloorDivOptions.FloorDivOptionsEnd(self._builder)

        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().FLOOR_DIV)
        if 'FLOOR_DIV' not in self._operatorCodeArray:
            self._operatorCodeArray['FLOOR_DIV'] = operatorCode
        return builtinOpTypes, builtinOp

    def builtin_REDUCE_ANY(self, op):
        pass

    def builtin_SQUARE(self, arg):
        builtinOpTypes = None
        builtinOp = None
        operatorCode = self._createOpcodeByIPUversion(tflite.BuiltinOperator.BuiltinOperator().SQUARE)
        if 'SQUARE' not in self._operatorCodeArray:
            self._operatorCodeArray['SQUARE'] = operatorCode
        return builtinOpTypes, builtinOp

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
        customName= ''
        if name == 'TFLite_CaffeSSD_Detection_PostProcess':#ssd postprocess
            #creat Operator Code
            customName = "TFLite_CaffeSSD_Detection_PostProcess"
            customCode = self._builder.CreateString(customName)
        elif name == 'TFLite_RFCSSD_Detection_PostProcess':#rfc postprocess
            #creat Operator Code
            customName = "TFLite_RFCSSD_Detection_PostProcess"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('tflite_detection_nms') == 0:
            #creat Operator Code
            customName = "TFLite_Detection_NMS"
            customCode = self._builder.CreateString(customName)
        elif name == 'TFLite_YoloV2_Detection_PostProcess' or name == 'TFLite_YoloV2_608_Detection_PostProcess':
            customName = "TFLite_YoloV2_Detection_PostProcess"
            customCode = self._builder.CreateString(customName)
        elif name == 'TFLite_YoloV3_Detection_PostProcess':
            customName = "TFLite_YoloV3_Detection_PostProcess"
            customCode = self._builder.CreateString(customName)
        elif name == 'TFLite_LanceNet_Detection_PostProcess':
            customName = "TFLite_LanceNet_Detection_PostProcess"
            customCode = self._builder.CreateString(customName)
        elif name == 'TFLite_FDA_Detection_PostProcess':
            customName = "TFLite_FDA_Detection_PostProcess"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('dilation') == 0:
            customName = "Dilation"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('sgs_lstm') == 0:
            customName = "SGS_LSTM"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('sgs_gru') == 0:
            customName = "SGS_GRU"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('postprocess_unpack') == 0:
            customName = "PostProcess_Unpack"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('image_concatenation') == 0:
            customName = "Image_Concatenation"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('sgs_roipooling') == 0:
            customName = "RoiPooling"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('erf') == 0:
            customName = "Erf"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('avepool3d') == 0:
            customName = "AvePool3D"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('maxpool3d') == 0:
            customName = "MaxPool3D"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('qirquantize') == 0:
            customName = "QIRQuantize"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('softplus') == 0:
            customName = "Softplus"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('customized_scatternd') == 0:
            customName = "Customized_ScatterND"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('groupconv') == 0:
            customName = "GroupConv"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('instancenorm') == 0:
            customName = "InstanceNorm"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('clip') == 0:
            customName = "Clip"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('custompow') == 0:
            customName = "CustomPow"
            customCode = self._builder.CreateString(customName)
        elif name.lower().find('atan') == 0:
            customName = "Atan"
            customCode = self._builder.CreateString(customName)
        else:
            arg = op.arg
            customCode = None
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == 'custom_opcode':
                customName = arg[i].str
                customCode = self._builder.CreateString(arg[i].str)
            mace_check(customCode is not None, "this Custom:%s layer will be support comming soon"%name)
        builtinOp = None
        builtinOpTypes = None
        #create Operator Code
        tflite.OperatorCode.OperatorCodeStart(self._builder)
        if getIPUVersion() == 'M6' or getIPUVersion() == 'I6E':
            tflite.OperatorCode.OperatorCodeAddDeprecatedBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().CUSTOM)
        else:
            tflite.OperatorCode.OperatorCodeAddBuiltinCode(self._builder, tflite.BuiltinOperator.BuiltinOperator().CUSTOM)
        tflite.OperatorCode.OperatorCodeAddCustomCode(self._builder,customCode)
        operatorCode = tflite.OperatorCode.OperatorCodeEnd(self._builder)
        if 'CUSTOM'+customName  not in self._operatorCodeArray:
            self._operatorCodeArray['CUSTOM'+customName] = operatorCode
        return builtinOpTypes,builtinOp,customName

def find_path(path, name):
    if path.split('/')[-1] == name:
        return path
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError('File `{}` not found in directory `{}`'.format(name, os.path.abspath(path)))
