import sys
from enum import Enum
import six
import pdb
import os
import struct
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
import numpy as np
import math
import tensorflow as tf
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.tools.graph_transforms import TransformGraph
from third_party import tflite
from third_party.crypto.vendor_crypto import *
from io import BytesIO

IS_PYTHON3 = sys.version_info > (3,)

TfliteSupportedOps = [
    'CONV_2D',
    'COS',
    'DEPTHWISE_CONV_2D',
    'RESHAPE',
    'SOFTMAX',
    'AVERAGE_POOL_2D',
    'MAX_POOL_2D',
    'RELU',
    'RELU6',
    'LEAKY_RELU',
    'TANH',
    'ADD',
    'CONCATENATION',
    'PAD',
    'FULLY_CONNECTED',
    'MEAN',
    'MUL',
    'SUB',
    'DIV',
    'TRANSPOSE',
    'SPLIT',
    'SPLIT_V',
    'SIN',
    'BATCH_TO_SPACE_ND',
    'SPACE_TO_BATCH_ND',
    'ARG_MAX',
    'RESIZE_BILINEAR',
    'LOGISTIC',
    'MAXIMUM',
    'MINIMUM',
    'CUSTOM',
    'EXP',
    'TRANSPOSE_CONV',
    'SLICE',
    'SQRT',
    'SQUARE',
    'RSQRT',
    'TILE',
    'PACK',
    'UNPACK',
    'STRIDED_SLICE',
    'BATCH_MATMUL',
    'ABS',
    'SUM',
    'RESIZE_NEAREST_NEIGHBOR',
    'PRELU',
    'CAST',
    'SHAPE',
    'LESS',
    'LOG',
    'GREATER',
    'GREATER_EQUAL',
    'EQUAL',
    'ROUND',
    'GATHER',
    'TOPKV2',
    'FLOOR_DIV'

]

TfliteOpType = Enum('TfliteOpType',
                  [(op, op) for op in TfliteSupportedOps],
                  type=str)

class TfliteConverter(base_converter.ConverterInterface):

    pooling_type_mode = {
        TfliteOpType.AVERAGE_POOL_2D.name: PoolingType.AVG,
        TfliteOpType.MAX_POOL_2D.name: PoolingType.MAX,
    }
    #padding_mode = {six.b(k): v for k, v in six.iteritems(padding_mode)}

    eltwise_type = {
        TfliteOpType.MUL.name: EltwiseType.PROD,
        TfliteOpType.ADD.name: EltwiseType.SUM,
        TfliteOpType.SUB.name: EltwiseType.SUB,
        TfliteOpType.DIV.name: EltwiseType.DIV,
        TfliteOpType.MAXIMUM.name: EltwiseType.MAX,
        TfliteOpType.MINIMUM.name: EltwiseType.MIN,
    }
    reduce_type = {
        TfliteOpType.MEAN.name: ReduceType.MEAN,
        TfliteOpType.SUM.name: ReduceType.SUM,
        #TfliteOpType.REDUCE_MAX.name: ReduceType.MAX,
        #TfliteOpType.REDUCE_MIN.name: ReduceType.MIN,
        #TfliteOpType.REDUCE_PROD.name: ReduceType.PROD,
        #TODO: REDUCE_ANY???
        #TfliteOpType.REDUCE_ANY.name:
    }
    activation_type = {
        TfliteOpType.RELU.name: ActivationType.RELU,
        TfliteOpType.PRELU.name: ActivationType.PRELU,
        TfliteOpType.RELU6.name: ActivationType.RELUX,
        TfliteOpType.TANH.name: ActivationType.TANH,
        TfliteOpType.LEAKY_RELU.name: ActivationType.LEAKYRELU,
        TfliteOpType.LOGISTIC.name: ActivationType.LOGISTIC,
    }

    def __init__(self, src_model_file, bFixed_model,model_buf=None,is_decrypt=False):
        self._op_converters = {
            TfliteOpType.AVERAGE_POOL_2D.name: self.convert_pooling,
            TfliteOpType.ADD.name: self.convert_eltwise,
            TfliteOpType.ARG_MAX.name: self.convert_argmax,
            TfliteOpType.ABS.name: self.convert_abs,
            TfliteOpType.BATCH_TO_SPACE_ND.name: self.convert_space_batch,
            TfliteOpType.BATCH_MATMUL.name: self.convert_batchmatmul,
            TfliteOpType.CAST.name: self.convert_cast,
            TfliteOpType.CONV_2D.name: self.convert_conv2d,
            TfliteOpType.COS.name: self.convert_cos,
            TfliteOpType.CONCATENATION.name: self.convert_concat,
            TfliteOpType.CUSTOM.name: self.convert_custom,
            TfliteOpType.DEPTHWISE_CONV_2D.name: self.convert_conv2d,
            TfliteOpType.DIV.name:self.convert_eltwise,
            TfliteOpType.EXP.name: self.convert_exp,
            TfliteOpType.EQUAL.name: self.convert_equal,
            TfliteOpType.FULLY_CONNECTED.name:self.convert_fullyconnected,
            TfliteOpType.GREATER.name:self.convert_greater,
            TfliteOpType.GREATER_EQUAL.name:self.convert_greaterequal,
            TfliteOpType.GATHER.name:self.convert_gather,
            TfliteOpType.LOGISTIC.name: self.convert_activation,
            TfliteOpType.LEAKY_RELU.name: self.convert_activation,
            TfliteOpType.LESS.name: self.convert_less,
            TfliteOpType.LOG.name: self.convert_log,
            TfliteOpType.PAD.name: self.convert_pad,
            TfliteOpType.PRELU.name: self.convert_activation,
            TfliteOpType.PACK.name: self.convert_pack,
            TfliteOpType.MUL.name:self.convert_eltwise,
            TfliteOpType.MINIMUM.name: self.convert_eltwise,
            TfliteOpType.MEAN.name:self.convert_reduce,
            TfliteOpType.MAXIMUM.name: self.convert_eltwise,
            TfliteOpType.MAX_POOL_2D.name: self.convert_pooling,
            TfliteOpType.RESHAPE.name: self.convert_reshape,
            TfliteOpType.RELU6.name: self.convert_activation,
            TfliteOpType.RELU.name: self.convert_activation,
            TfliteOpType.RELU6.name: self.convert_activation,
            TfliteOpType.RESIZE_BILINEAR.name: self.convert_resize_bilinear,
            TfliteOpType.RSQRT.name: self.convert_rsqrt,
            TfliteOpType.RESIZE_NEAREST_NEIGHBOR.name: self.convert_resize_nearest_neighbor,
            TfliteOpType.ROUND.name: self.convert_round,
            TfliteOpType.SUM.name:self.convert_reduce,
            TfliteOpType.SPLIT.name:self.convert_split,
            TfliteOpType.SPLIT_V.name:self.convert_split_V,
            TfliteOpType.SPACE_TO_BATCH_ND.name: self.convert_space_batch,
            TfliteOpType.SOFTMAX.name: self.convert_softmax,
            TfliteOpType.SUB.name:self.convert_eltwise,
            TfliteOpType.SLICE.name: self.convert_slice,
            TfliteOpType.SQRT.name: self.convert_sqrt,
            TfliteOpType.SQUARE.name: self.convert_square,
            TfliteOpType.SIN.name: self.convert_sin,
            TfliteOpType.STRIDED_SLICE.name: self.convert_stridedslice,
            TfliteOpType.SHAPE.name: self.convert_shape,
            TfliteOpType.TRANSPOSE.name:self.convert_transpose,
            TfliteOpType.TRANSPOSE_CONV.name: self.convert_transpose_conv,
            TfliteOpType.TANH.name: self.convert_activation,
            TfliteOpType.TILE.name: self.convert_tile,
            TfliteOpType.TOPKV2.name: self.convert_topkv2,
            TfliteOpType.FLOOR_DIV.name: self.convert_floordiv,
            TfliteOpType.UNPACK.name: self.convert_unpack,

        }
        self._mace_net_def = mace_pb2.NetDef()
        self._data_format = mace_pb2.DT_NHWC
        self._input_name = []
        self._output_name = []
        self.input_node_shapes_array = []
        self._consts = {}
        self._bFixed_model = bFixed_model

        # import tflite graph
        if model_buf is None:
            with open(src_model_file, 'rb') as f:
                buf = f.read()
                model = tflite.Model.Model.GetRootAsModel(buf, 0)
                self._model = model
                tensor_name_mapping = dict()
                data_format_arg = self._mace_net_def.arg.add()
                data_format_arg.name = 'subgraph_name'

                for subgraph_idx in range(model.SubgraphsLength()):
                    subgraph = model.Subgraphs(subgraph_idx)
                    if subgraph.Name() is not None:
                        data_format_arg.str = subgraph.Name()
                    print(f'[{subgraph_idx}] Subgraph "{subgraph.Name()}" contains {subgraph.TensorsLength()} tensors:')
                    for tensor_idx in range(subgraph.TensorsLength()):
                        tensor = subgraph.Tensors(tensor_idx)
                        shape = tensor.ShapeAsNumpy()
                        name = "".join(map(chr, tensor.Name()))
                        tensor_name_mapping[tensor_idx] = name
                        buffer_idx = tensor.Buffer()
                        buffer = model.Buffers(buffer_idx)
                        print(f'[TensorID:{tensor_idx}] Tensor "{name}"  has shape {shape} buffer_length:[{buffer.DataLength()}] ')
                    #Get input name,shape
                    for input_idx in range(subgraph.InputsLength()):
                        tensorId = subgraph.Inputs(input_idx)
                        tensor = subgraph.Tensors(tensorId)
                        name = "".join(map(chr, tensor.Name()))
                        self._input_name.append(name)
                        self.input_node_shapes_array.append(list(tensor.ShapeAsNumpy()))
                    #Get output name,shape
                    for Output_idx in range(subgraph.OutputsLength()):
                        tensorId = subgraph.Outputs(Output_idx)
                        tensor = subgraph.Tensors(tensorId)
                        name = "".join(map(chr, tensor.Name()))
                        self._output_name.append(name)
                    print(f'The subgraph has {subgraph.OperatorsLength()} operators.')
                    for op_idx in range(subgraph.OperatorsLength()):
                        op = subgraph.Operators(op_idx)
                        op_code = model.OperatorCodes(op.OpcodeIndex())
                        #If schema version < 2.4 using DeprecatedBuiltinCode() to get builtinOP enum value
                        #If schema version >= 2.4 using BuiltinCode() to get builtinOP enum value
                        optype = self.tflite_getType(tflite.BuiltinOperator.BuiltinOperator,
                                                    max(op_code.DeprecatedBuiltinCode(),op_code.BuiltinCode()))
                        print(f'[{op_idx}] Operator of type "{optype}" has:')
                        # BuiltinOptions(op.BuiltinOptionsType())
                        for (name, item) in [('Inputs', op.InputsAsNumpy()), ('Outputs', op.OutputsAsNumpy())]:
                            print(f'{name}: [{" -> ".join(f"{tensor_name_mapping[node]}({node})" for node in item)}]')
        else:
            if is_decrypt:
                decrypt_model_buf = decrypt(model_buf)
                model_BytesIO = decrypt_model_buf.read()
                model = tflite.Model.Model.GetRootAsModel(model_BytesIO, 0)
            else:
                model_BytesIO = model_buf.read()
                model = tflite.Model.Model.GetRootAsModel(model_BytesIO, 0)
            self._model = model
            tensor_name_mapping = dict()
            data_format_arg = self._mace_net_def.arg.add()
            data_format_arg.name = 'subgraph_name'

            for subgraph_idx in range(model.SubgraphsLength()):
                subgraph = model.Subgraphs(subgraph_idx)
                if subgraph.Name() is not None:
                    data_format_arg.str = subgraph.Name()
                print(f'[{subgraph_idx}] Subgraph "{subgraph.Name()}" contains {subgraph.TensorsLength()} tensors:')
                for tensor_idx in range(subgraph.TensorsLength()):
                    tensor = subgraph.Tensors(tensor_idx)
                    shape = tensor.ShapeAsNumpy()
                    name = "".join(map(chr, tensor.Name()))
                    tensor_name_mapping[tensor_idx] = name
                    buffer_idx = tensor.Buffer()
                    buffer = model.Buffers(buffer_idx)
                    print(f'[TensorID:{tensor_idx}] Tensor "{name}"  has shape {shape} buffer_length:[{buffer.DataLength()}] ')
                #Get input name,shape
                for input_idx in range(subgraph.InputsLength()):
                    tensorId = subgraph.Inputs(input_idx)
                    tensor = subgraph.Tensors(tensorId)
                    name = "".join(map(chr, tensor.Name()))
                    self._input_name.append(name)
                    self.input_node_shapes_array.append(list(tensor.ShapeAsNumpy()))
                #Get output name,shape
                for Output_idx in range(subgraph.OutputsLength()):
                    tensorId = subgraph.Outputs(Output_idx)
                    tensor = subgraph.Tensors(tensorId)
                    name = "".join(map(chr, tensor.Name()))
                    self._output_name.append(name)
                print(f'The subgraph has {subgraph.OperatorsLength()} operators.')
                for op_idx in range(subgraph.OperatorsLength()):
                    op = subgraph.Operators(op_idx)
                    op_code = model.OperatorCodes(op.OpcodeIndex())
                    #If schema version < 2.4 using DeprecatedBuiltinCode() to get builtinOP enum value
                    #If schema version >= 2.4 using BuiltinCode() to get builtinOP enum value
                    optype = self.tflite_getType(tflite.BuiltinOperator.BuiltinOperator,
                                                max(op_code.DeprecatedBuiltinCode(),op_code.BuiltinCode()))
                    print(f'[{op_idx}] Operator of type "{optype}" has:')
                    # BuiltinOptions(op.BuiltinOptionsType())
                    for (name, item) in [('Inputs', op.InputsAsNumpy()), ('Outputs', op.OutputsAsNumpy())]:
                        print(f'{name}: [{" -> ".join(f"{tensor_name_mapping[node]}({node})" for node in item)}]')

    def convert_ops(self):
        for subgraph_idx in range(self._model.SubgraphsLength()):
            subgraph = self._model.Subgraphs(subgraph_idx)
            for op_idx in range(subgraph.OperatorsLength()):
                #Get operator
                op = subgraph.Operators(op_idx)
                op_code = self._model.OperatorCodes(op.OpcodeIndex())
                optype = self.tflite_getType(tflite.BuiltinOperator.BuiltinOperator,
                                            max(op_code.DeprecatedBuiltinCode(),op_code.BuiltinCode()))
                mace_check(optype in self._op_converters,
                       "Mace does not support tensorflow op type %s yet"
                       % optype)
                self._op_converters[optype](op, optype, subgraph)

    def tflite_getType(self, CLASS, code):
        for name, value in CLASS.__dict__.items():
            if value == code:
                return name
        return None

    def convertToMaceDataType(self, type):
        maceTensorDataType = 0
        if type == 'UINT8':
            maceTensorDataType = mace_pb2.DT_UINT8
        elif type == 'INT8':
            maceTensorDataType = mace_pb2.DT_INT8
        elif type == 'INT32':
            maceTensorDataType = mace_pb2.DT_INT32
        else:
            maceTensorDataType = mace_pb2.DT_FLOAT
        return maceTensorDataType

    def convert_tensors(self):
        bFixed_model = self._bFixed_model
        for subgraph_idx in range(self._model.SubgraphsLength()):
            subgraph = self._model.Subgraphs(subgraph_idx)
            for tenosr_idx in range(subgraph.TensorsLength()):
                #add tensor to maceAttr
                mace_tensor = self._mace_net_def.tensors.add()
                tfliteTensor = subgraph.Tensors(tenosr_idx)
                mace_tensor.name = tfliteTensor.Name()
                #set tensor dim
                tfliteTensor_ShapeAsNumpy = tfliteTensor.ShapeAsNumpy()
                mace_tensor.dims.extend(list(tfliteTensor_ShapeAsNumpy))
                mace_tensor.data_format = self._data_format
                # Get the tensor buffer
                tfliteTensor_Buffer = self._model.Buffers(tfliteTensor.Buffer())
                tfliteTensor_BufferAsNumpy = tfliteTensor_Buffer.DataAsNumpy()
                tfliteTensor_dataType = self.tflite_getType(tflite.TensorType.TensorType, tfliteTensor.Type())
                if tfliteTensor_dataType == 'INT64':
                    #deep_lab_v3 Arg_Max output is int64 but maceTensor unsupport so transform to float32
                    mace_tensor.data_type = tflite.TensorType.TensorType.FLOAT32
                else:
                    #mace_tensor.data_type = tfliteTensor.Type()
                    mace_tensor.data_type = self.convertToMaceDataType(tfliteTensor_dataType)

                #set const tensor buffer
                if(tfliteTensor_Buffer.DataLength() != 0):
                    if tfliteTensor_dataType == "FLOAT32":
                        mace_tensor.data_type = mace_pb2.DT_FLOAT
                        #convert buffer data type from uint8
                        tfliteTensor_Bytes = struct.pack('B' * (tfliteTensor_Buffer.DataLength()),
                            *list(tfliteTensor_BufferAsNumpy))
                        tfliteTensor_BufferAsNumpy = struct.unpack('f' * (len(tfliteTensor_Bytes) // 4), tfliteTensor_Bytes)
                        mace_tensor.float_data.extend(tfliteTensor_BufferAsNumpy)
                    elif tfliteTensor_dataType == "INT32":
                        mace_tensor.data_type = mace_pb2.DT_INT32
                        #convert buffer data type from uint8
                        tfliteTensor_Bytes = struct.pack('B' * (tfliteTensor_Buffer.DataLength()),
                            *list(tfliteTensor_BufferAsNumpy))
                        tfliteTensor_BufferAsNumpy = struct.unpack('i' * (len(tfliteTensor_Bytes) // 4), tfliteTensor_Bytes)
                        mace_tensor.int32_data.extend(tfliteTensor_BufferAsNumpy)
                    elif tfliteTensor_dataType == "INT8":
                        mace_tensor.data_type = mace_pb2.DT_INT8
                        #convert buffer data type from uint8
                        tfliteTensor_Bytes = struct.pack('B' * (tfliteTensor_Buffer.DataLength()),
                            *list(tfliteTensor_BufferAsNumpy))
                        tfliteTensor_BufferAsNumpy = struct.unpack('b' * (len(tfliteTensor_Bytes)), tfliteTensor_Bytes)
                        mace_tensor.int32_data.extend(tfliteTensor_BufferAsNumpy)
                    elif tfliteTensor_dataType == "UINT8":
                        mace_tensor.data_type = mace_pb2.DT_UINT8
                        #convert buffer data type from uint8
                        tfliteTensor_Bytes = struct.pack('B' * (tfliteTensor_Buffer.DataLength()),
                            *list(tfliteTensor_BufferAsNumpy))
                        tfliteTensor_BufferAsNumpy = struct.unpack('B' * (len(tfliteTensor_Bytes)), tfliteTensor_Bytes)
                        mace_tensor.int32_data.extend(tfliteTensor_BufferAsNumpy)
                    else:
                        mace_check(False,
                                   "MaceTensor Not supported tensor type: %s" % tfliteTensor_dataType)
                #set the tensor quantization info if tflite is a fixed_model
                if(bFixed_model == True):
                    if(tfliteTensor.Quantization() != None):
                        quant = tfliteTensor.Quantization()
                        if(quant.MaxLength() > 0):
                            MaxAsNumpy = quant.MaxAsNumpy()
                            mace_tensor.maxval.extend(list(MaxAsNumpy))
                        if(quant.MinLength() > 0):
                            MinAsNumpy = quant.MinAsNumpy()
                            mace_tensor.minval.extend(list(MinAsNumpy))
                        if(quant.ScaleLength() > 0):
                            ScaleAsNumpy = quant.ScaleAsNumpy()
                            mace_tensor.scale.extend(list(ScaleAsNumpy))
                        if(quant.ZeroPointLength() > 0):
                            ZeroPointAsNumpy = quant.ZeroPointAsNumpy()
                            mace_tensor.zero_point.extend(list(ZeroPointAsNumpy))
                ###TODO using 2.7 tflite model verify under the code flow###
                if(tfliteTensor.Sparsity() != None):
                    SparsityTensor = tfliteTensor.Sparsity()
                    if(SparsityTensor.TraversalOrderLength() > 0):
                        TraversalOrderAsNumpy = SparsityTensor.TraversalOrderAsNumpy()
                        mace_tensor.sparsity.traversal_order.extend(list(TraversalOrderAsNumpy))
                    if(SparsityTensor.BlockMapLength() > 0):
                        BlockMapAsNumpy = SparsityTensor.BlockMapAsNumpy()
                        mace_tensor.sparsity.block_map.extend(list(BlockMapAsNumpy))
                    if(SparsityTensor.DimMetadataLength() > 0):
                        for index in range(SparsityTensor.DimMetadataLength()):
                            mace_dimMetadata = mace_tensor.sparsity.dim_metadata.add()
                            DimMetadata = SparsityTensor.DimMetadata(index)
                            mace_dimMetadata_format = self.tflite_getType(tflite.DimensionType.DimensionType, DimMetadata.Format())
                            if mace_dimMetadata_format == 'DENSE':
                                #If format is DENSE then we use the dense_size field to
                                #store the size of that dimension.
                                mace_dimMetadata.format = 0
                                mace_dimMetadata.dense_size = DimMetadata.DenseSize()
                            elif mace_dimMetadata_format == 'SPARSE_CSR':
                                #If format is SPARSE_CSR then we use array_segments and
                                #array_indices to encode that dimension.
                                #mace_dimMetadata.array_indices = DimMetadata.ArrayIndices()
                                #mace_dimMetadata.array_segments = DimMetadata.ArraySegments()
                                mace_dimMetadata.format = 1
                                arrayIndices_type = self.tflite_getType(tflite.SparseIndexVector.SparseIndexVector,
                                                                        DimMetadata.ArrayIndicesType())
                                ArrayIndices = DimMetadata.ArrayIndices()
                                if arrayIndices_type == 'Int32Vector':
                                    mace_dimMetadata.ArrayIndicesVecType = 'Int32Vector'
                                    IndexVector = tflite.Int32Vector.Int32Vector()
                                    IndexVector.Init(ArrayIndices.Bytes, ArrayIndices.Pos)
                                    IndicesValuesAsNumpy = IndexVector.ValuesAsNumpy()
                                    mace_dimMetadata.array_indices.s32Vector.values.extend(list(IndicesValuesAsNumpy))
                                elif arrayIndices_type == 'Uint16Vector':
                                    mace_dimMetadata.ArrayIndicesVecType = 'Uint16Vector'
                                    IndexVector = tflite.Uint16Vector.Uint16Vector()
                                    IndexVector.Init(ArrayIndices.Bytes, ArrayIndices.Pos)
                                    IndicesValuesAsNumpy = IndexVector.ValuesAsNumpy()
                                    mace_dimMetadata.array_indices.u16Vector.values.extend(list(IndicesValuesAsNumpy))

                                elif arrayIndices_type == 'Uint8Vector':
                                    mace_dimMetadata.ArrayIndicesVecType = 'Uint8Vector'
                                    IndexVector = tflite.Uint8Vector.Uint8Vector()
                                    IndexVector.Init(ArrayIndices.Bytes, ArrayIndices.Pos)
                                    IndicesValuesAsNumpy = IndexVector.ValuesAsNumpy()
                                    mace_dimMetadata.array_indices.u8Vector.values.extend(list(IndicesValuesAsNumpy))
                                else:
                                    mace_check(False,
                                        "Unsupported ArrayIndices type: %s" % arrayIndices_type)

                                arraySegments_type = self.tflite_getType(tflite.SparseIndexVector.SparseIndexVector,
                                           DimMetadata.ArraySegmentsType())
                                ArraySegments = DimMetadata.ArraySegments()
                                if arraySegments_type == 'Int32Vector':
                                    mace_dimMetadata.ArraySegmentsVecType = 'Int32Vector'
                                    IndexVector = tflite.Int32Vector.Int32Vector()
                                    IndexVector.Init(ArraySegments.Bytes, ArraySegments.Pos)
                                    SegmentsValuesAsNumpy = IndexVector.ValuesAsNumpy()
                                    mace_dimMetadata.array_segments.s32Vector.values.extend(list(SegmentsValuesAsNumpy))
                                elif arraySegments_type == 'Uint16Vector':
                                    mace_dimMetadata.ArraySegmentsVecType = 'Uint16Vector'
                                    IndexVector = tflite.Uint16Vector.Uint16Vector()
                                    IndexVector.Init(ArraySegments.Bytes, ArraySegments.Pos)
                                    SegmentsValuesAsNumpy = IndexVector.ValuesAsNumpy()
                                    mace_dimMetadata.array_segments.u16Vector.values.extend(list(SegmentsValuesAsNumpy))
                                elif arraySegments_type == 'Uint8Vector':
                                    mace_dimMetadata.ArraySegmentsVecType = 'Uint8Vector'
                                    IndexVector = tflite.Uint8Vector.Uint8Vector()
                                    IndexVector.Init(ArraySegments.Bytes, ArraySegments.Pos)
                                    SegmentsValuesAsNumpy = IndexVector.ValuesAsNumpy()
                                    mace_dimMetadata.array_segments.u8Vector.values.extend(list(SegmentsValuesAsNumpy))
                                else:
                                    mace_check(False,
                                        "Unsupported ArraySegmentsType type: %s" % arraySegments_type)
                            else:
                                mace_check(False,
                                        "Unsupported format type: %s" % mace_dimMetadata.format)
                if(tfliteTensor.ShapeSignatureLength() > 0):
                    ShapeSignatureAsNumpy = tfliteTensor.ShapeSignatureAsNumpy()
                    mace_tensor.shape_signature.extend(list(ShapeSignatureAsNumpy))
                self._consts[mace_tensor.name] = mace_tensor

    def run(self):
        self.convert_tensors()
        self.convert_ops()
        for name in self._input_name:
            input_info = self._mace_net_def.input_info.add()
            input_info.name = name
        for name in self._output_name:
            output_info = self._mace_net_def.output_info.add()
            output_info.name = name
        return self._mace_net_def, self._input_name, self._output_name

    def convert_general_op(self, tflite_op, op_type, subgraph):
        mace_op = self._mace_net_def.op.add()
        mace_op.name = str(tflite_op.OpcodeIndex())
        mace_op.type = op_type
        IntermediatesAsNumpy = tflite_op.IntermediatesAsNumpy()
        mace_op.intermediates.extend(IntermediatesAsNumpy)
        data_format_arg = mace_op.arg.add()
        data_format_arg.name = MaceKeyword.mace_data_format_str
        data_format_arg.i = self._data_format

        for input_idx in range(tflite_op.InputsLength()):
            input_tensor_index = tflite_op.Inputs(input_idx)
            input_tensor = subgraph.Tensors(input_tensor_index)
            #input_tensor.Name().decode('utf-8')
            input_tensor_name = "".join(map(chr, input_tensor.Name()))
            mace_op.input.extend([input_tensor_name])

        for output_idx in range(tflite_op.OutputsLength()):
            output_tensor_index = tflite_op.Outputs(output_idx)
            output_tensor = subgraph.Tensors(output_tensor_index)
            #input_tensor.Name().decode('utf-8')
            output_shapeAsNp = output_tensor.ShapeAsNumpy()
            output_tensor_name = "".join(map(chr, output_tensor.Name()))
            mace_op.output.extend([output_tensor_name])
            output_shape = mace_op.output_shape.add()

            output_shape.dims.extend(list(output_shapeAsNp))
            #output_shape.dims.extend(self.Dict_Output_name_shape[output_tensor_name])
            framework_type_arg = mace_op.arg.add()
            framework_type_arg.name = MaceKeyword.mace_framework_type_str
            framework_type_arg.i = FrameworkType.TENSORFLOW.value
        return mace_op

    def convert_conv2d(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        if op_type == TfliteOpType.CONV_2D.name:
            mace_op.type = MaceOp.Conv2D.name
            opt = tflite.Conv2DOptions.Conv2DOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
        elif op_type == TfliteOpType.DEPTHWISE_CONV_2D.name:
            mace_op.type = MaceOp.DepthwiseConv2d.name
            opt = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
        else:
            #TODO: check if tflite support DepthwiseConv2dNative,Conv2DBackpropInput
            print("tflite unsupport optype")
        padding_type = opt.Padding()
        strides_hw = [opt.StrideH(), opt.StrideW()]
        dilation_hw = [opt.DilationHFactor(), opt.DilationWFactor()]
        padding_value = [opt.PaddingLeft(), opt.PaddingRight(), opt.PaddingTop(), opt.PaddingBottom()]
        fusedActivationFunction_str = self.tflite_getType(tflite.ActivationFunctionType.ActivationFunctionType,opt.FusedActivationFunction())
        fusedActivationFunction = opt.FusedActivationFunction()
        #set fusedActivationFunction
        if fusedActivationFunction_str == 'RELU6':
            relu6_arg = mace_op.arg.add()
            relu6_arg.name = fusedActivationFunction_str
            relu6_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'RELU':
            relu_arg = mace_op.arg.add()
            relu_arg.name = fusedActivationFunction_str
            relu_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'RELU_N1_TO_1':
            relu_arg = mace_op.arg.add()
            relu_arg.name = fusedActivationFunction_str
            relu_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'TANH':
            tabh_arg = mace_op.arg.add()
            tabh_arg.name = fusedActivationFunction_str
            tabh_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'SIGN_BIT':
            sign_arg = mace_op.arg.add()
            sign_arg.name = fusedActivationFunction_str
            sign_arg.i = fusedActivationFunction
        #set padding_value
        padding_arg = mace_op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_values_str
        padding_arg.ints.extend(padding_value)
        #set padding_type
        padding_arg = mace_op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = padding_type
        #set strides
        strides_arg = mace_op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(strides_hw)
        #set dilation
        dilation_arg = mace_op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        try:
            dilation_hw = [opt.DilationHFactor(), opt.DilationWFactor()]
        except ValueError:
            dilation_hw = [1, 1]
        dilation_arg.ints.extend(dilation_hw)
        return

    def convert_activation(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Activation.name
        type_arg = mace_op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b(self.activation_type[op_type].name)

        if op_type == TfliteOpType.RELU6.name:
            limit_arg = mace_op.arg.add()
            limit_arg.name = MaceKeyword.mace_activation_max_limit_str
            limit_arg.f = 6.0
        elif op_type == TfliteOpType.LEAKY_RELU.name:
            #Get option
            op_opt = tflite_op.BuiltinOptions()
            opt = tflite.LeakyReluOptions.LeakyReluOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            LeakyRelu_Alpha = opt.Alpha()
            alpha_arg = mace_op.arg.add()
            alpha_arg.name = \
            MaceKeyword.mace_activation_leakyrelu_coefficient_str
            alpha_arg.f = LeakyRelu_Alpha

    def convert_pooling(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Pooling.name
        pooling_type_arg = mace_op.arg.add()
        pooling_type_arg.name = MaceKeyword.mace_pooling_type_str
        if op_type == 'AVERAGE_POOL_2D':
            pooling_type_arg.i = self.pooling_type_mode[TfliteOpType.AVERAGE_POOL_2D.name].value
        elif(op_type == 'MAX_POOL_2D'):
            pooling_type_arg.i = self.pooling_type_mode[TfliteOpType.MAX_POOL_2D.name].value
        else:
            mace_check(False, "op_type has unsupport {op_type} type ")
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.Pool2DOptions.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        padding_type = opt.Padding()
        strides_hw = [opt.StrideH(), opt.StrideW()]
        filter_hw = [opt.FilterHeight(), opt.FilterWidth()]
        padding_value = [opt.PaddingLeft(), opt.PaddingRight(), opt.PaddingTop(), opt.PaddingBottom()]
        fusedActivationFunction_str = self.tflite_getType(tflite.ActivationFunctionType.ActivationFunctionType,opt.FusedActivationFunction())
        fusedActivationFunction = opt.FusedActivationFunction()
        #set fusedActivationFunction
        if fusedActivationFunction_str == 'RELU6':
            relu6_arg = mace_op.arg.add()
            relu6_arg.name = fusedActivationFunction_str
            relu6_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'RELU':
            relu_arg = mace_op.arg.add()
            relu_arg.name = fusedActivationFunction_str
            relu_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'RELU_N1_TO_1':
            relu_arg = mace_op.arg.add()
            relu_arg.name = fusedActivationFunction_str
            relu_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'TANH':
            tabh_arg = mace_op.arg.add()
            tabh_arg.name = fusedActivationFunction_str
            tabh_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'SIGN_BIT':
            sign_arg = mace_op.arg.add()
            sign_arg.name = fusedActivationFunction_str
            sign_arg.i = fusedActivationFunction
        #set padding_value
        padding_arg = mace_op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_values_str
        padding_arg.ints.extend(padding_value)
        #set padding_type
        padding_arg = mace_op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = padding_type
        #set strides
        strides_arg = mace_op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(strides_hw)
        #set filters
        filters_arg = mace_op.arg.add()
        filters_arg.name = MaceKeyword.mace_kernel_str
        filters_arg.ints.extend(filter_hw)

    def convert_reshape(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        if op_type == TfliteOpType.RESHAPE.name:
            mace_op.type = MaceOp.Reshape.name
        else:
            mace_check(False, "op_type has invalid type ")
        #Get option
        #op_opt = tflite_op.BuiltinOptions()
        #opt = tflite.ReshapeOptions.ReshapeOptions()
        #opt.Init(op_opt.Bytes, op_opt.Pos)
        #new_shape = opt.NewShapeAsNumpy()

    def convert_softmax(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        if op_type == TfliteOpType.SOFTMAX.name:
            mace_op.type = MaceOp.Softmax.name
        else:
            mace_check(False, "op_type has invalid type ")
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.SoftmaxOptions.SoftmaxOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        beta = opt.Beta()
        #set beta
        beta_arg = mace_op.arg.add()
        beta_arg.name = "beta"
        beta_arg.floats.append(beta)

    def convert_eltwise(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Eltwise.name
        type_arg = mace_op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = self.eltwise_type[op_type].value
        #TODO: add eltwise op flow
        if op_type == TfliteOpType.ADD.name:
            #Get option
            op_opt = tflite_op.BuiltinOptions()
            opt = tflite.AddOptions.AddOptions()
            if op_opt is not None:
                opt.Init(op_opt.Bytes, op_opt.Pos)
                #set PotScaleInt16
                PotScaleInt16 = opt.PotScaleInt16()
                PotScaleInt16_arg = mace_op.arg.add()
                PotScaleInt16_arg.name = MaceKeyword.mace_PotScaleInt16_str
                PotScaleInt16_arg.i = PotScaleInt16

        elif op_type == TfliteOpType.SUB.name:
            #Get option
            op_opt = tflite_op.BuiltinOptions()
            opt = tflite.SubOptions.SubOptions()
            if op_opt is not None:
                opt.Init(op_opt.Bytes, op_opt.Pos)
                #set PotScaleInt16
                PotScaleInt16 = opt.PotScaleInt16()
                PotScaleInt16_arg = mace_op.arg.add()
                PotScaleInt16_arg.name = MaceKeyword.mace_PotScaleInt16_str
                PotScaleInt16_arg.i = PotScaleInt16
        elif op_type == TfliteOpType.MUL.name:
            #Get option
            op_opt = tflite_op.BuiltinOptions()
            opt = tflite.MulOptions.MulOptions()
        elif op_type == TfliteOpType.DIV.name:
            #Get option
            op_opt = tflite_op.BuiltinOptions()
            opt = tflite.DivOptions.DivOptions()
        elif op_type == TfliteOpType.MAXIMUM.name or op_type == TfliteOpType.MINIMUM.name:
            #MAXIMUM without option
            return
        #GET & SET fusedActivationFunction
        if op_opt is not None:
            opt.Init(op_opt.Bytes, op_opt.Pos)
            fusedActivationFunction = opt.FusedActivationFunction()
            fusedActivationFunction_str = self.tflite_getType(tflite.ActivationFunctionType.ActivationFunctionType, opt.FusedActivationFunction())
            if fusedActivationFunction_str == 'RELU6':
                relu6_arg = mace_op.arg.add()
                relu6_arg.name = fusedActivationFunction_str
                relu6_arg.i = fusedActivationFunction
            elif fusedActivationFunction_str == 'RELU':
                relu_arg = mace_op.arg.add()
                relu_arg.name = fusedActivationFunction_str
                relu_arg.i = fusedActivationFunction
            elif fusedActivationFunction_str == 'RELU_N1_TO_1':
                relu_arg = mace_op.arg.add()
                relu_arg.name = fusedActivationFunction_str
                relu_arg.i = fusedActivationFunction
            elif fusedActivationFunction_str == 'TANH':
                tabh_arg = mace_op.arg.add()
                tabh_arg.name = fusedActivationFunction_str
                tabh_arg.i = fusedActivationFunction
            elif fusedActivationFunction_str == 'SIGN_BIT':
                sign_arg = mace_op.arg.add()
                sign_arg.name = fusedActivationFunction_str
                sign_arg.i = fusedActivationFunction

    def convert_concat(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Concat.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.ConcatenationOptions.ConcatenationOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set axis_value
        axis_value = opt.Axis()
        FusedActivationFunction = opt.FusedActivationFunction()
        axis_arg = mace_op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value
        #GET & SET fusedActivationFunction
        fusedActivationFunction = opt.FusedActivationFunction()
        fusedActivationFunction_str = self.tflite_getType(tflite.ActivationFunctionType.ActivationFunctionType, opt.FusedActivationFunction())
        if fusedActivationFunction_str == 'RELU6':
            relu6_arg = mace_op.arg.add()
            relu6_arg.name = fusedActivationFunction_str
            relu6_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'RELU':
            relu_arg = mace_op.arg.add()
            relu_arg.name = fusedActivationFunction_str
            relu_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'RELU_N1_TO_1':
            relu_arg = mace_op.arg.add()
            relu_arg.name = fusedActivationFunction_str
            relu_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'TANH':
            tabh_arg = mace_op.arg.add()
            tabh_arg.name = fusedActivationFunction_str
            tabh_arg.i = fusedActivationFunction
        elif fusedActivationFunction_str == 'SIGN_BIT':
            sign_arg = mace_op.arg.add()
            sign_arg.name = fusedActivationFunction_str
            sign_arg.i = fusedActivationFunction

    def convert_pad(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Pad.name

    def convert_fullyconnected(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.FullyConnected.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.FullyConnectedOptions.FullyConnectedOptions()
        if opt is not None:
            opt.Init(op_opt.Bytes, op_opt.Pos)
            #set WeightsFormat
            WeightsFormat = opt.WeightsFormat()
            WeightsFormat_str = self.tflite_getType(tflite.FullyConnectedOptionsWeightsFormat.FullyConnectedOptionsWeightsFormat, opt.WeightsFormat())
            if WeightsFormat_str == 'SHUFFLED4x16INT8':
                format_arg = mace_op.arg.add()
                format_arg.name = WeightsFormat_str
                format_arg.i = WeightsFormat
            #set FusedActivationFunction
            FusedActivationFunction = opt.FusedActivationFunction()
            fusedActivationFunction_str = self.tflite_getType(tflite.ActivationFunctionType.ActivationFunctionType, opt.FusedActivationFunction())
            if fusedActivationFunction_str == 'RELU6':
                relu6_arg = mace_op.arg.add()
                relu6_arg.name = fusedActivationFunction_str
                relu6_arg.i = FusedActivationFunction
            elif fusedActivationFunction_str == 'RELU':
                relu_arg = mace_op.arg.add()
                relu_arg.name = fusedActivationFunction_str
                relu_arg.i = FusedActivationFunction
            elif fusedActivationFunction_str == 'RELU_N1_TO_1':
                relu_arg = mace_op.arg.add()
                relu_arg.name = fusedActivationFunction_str
                relu_arg.i = FusedActivationFunction
            elif fusedActivationFunction_str == 'TANH':
                tabh_arg = mace_op.arg.add()
                tabh_arg.name = fusedActivationFunction_str
                tabh_arg.i = FusedActivationFunction
            elif fusedActivationFunction_str == 'SIGN_BIT':
                sign_arg = mace_op.arg.add()
                sign_arg.name = fusedActivationFunction_str
                sign_arg.i = FusedActivationFunction
            #set KeepNumDims
            KeepNumDims = opt.KeepNumDims()
            KeepNumDims_arg = mace_op.arg.add()
            KeepNumDims_arg.name = 'KeepNumDims'
            KeepNumDims_arg.i = KeepNumDims
            #set AsymmetricQuantizeInputs
            AsymmetricQuantizeInputs = opt.AsymmetricQuantizeInputs()
            AsymmetricQuantizeInputs_arg = mace_op.arg.add()
            AsymmetricQuantizeInputs_arg.name = MaceKeyword.mace_AsymmetricQuantizeInputs_str
            AsymmetricQuantizeInputs_arg.i = AsymmetricQuantizeInputs


    def convert_reduce(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Reduce.name

        reduce_type_arg = mace_op.arg.add()
        reduce_type_arg.name = MaceKeyword.mace_reduce_type_str
        reduce_type_arg.i = self.reduce_type[op_type].value
        # if op_type == 'MEAN' or op_type == "SUM":
        #     #MEAN_OP not have option
        #     return
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.ReducerOptions.ReducerOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set KeepDims
        KeepDims = opt.KeepDims()
        keep_dims_arg = mace_op.arg.add()
        keep_dims_arg.name = MaceKeyword.mace_keepdims_str
        keep_dims_arg.i = KeepDims

    def convert_transpose(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Transpose.name
    def convert_cast(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Cast.name
    def convert_split(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Split.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.SplitOptions.SplitOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set NumSplits
        NumSplits = opt.NumSplits()
        NumSplits_arg = mace_op.arg.add()
        NumSplits_arg.name = MaceKeyword.mace_num_split_str
        NumSplits_arg.i = NumSplits

    def convert_split_V(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Split_V.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.SplitVOptions.SplitVOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set NumSplits
        NumSplits = opt.NumSplits()
        NumSplits_arg = mace_op.arg.add()
        NumSplits_arg.name = MaceKeyword.mace_num_split_str
        NumSplits_arg.i = NumSplits

    def convert_space_batch(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        if op_type == TfliteOpType.BATCH_TO_SPACE_ND.name:
            mace_op.type = MaceOp.BatchToSpaceND.name
        else:
            #SPACE_TO_BATCH_ND
            mace_op.type = MaceOp.SpaceToBatchND.name

    def convert_argmax(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.ArgMax.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.ArgMaxOptions.ArgMaxOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set OutputType
        OutputType = opt.OutputType()
        OutputType_arg = mace_op.arg.add()
        OutputType_arg.name = MaceKeyword.mace_output_type_str
        OutputType_arg.i = OutputType

    def convert_resize_bilinear(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.ResizeBilinear.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.ResizeBilinearOptions.ResizeBilinearOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set AlignCorners
        AlignCorners = opt.AlignCorners()
        AlignCorners_arg = mace_op.arg.add()
        AlignCorners_arg.name = MaceKeyword.mace_align_corners_str
        AlignCorners_arg.i = AlignCorners
        #set HalfPixelCenters
        HalfPixelCenters = opt.HalfPixelCenters()
        if HalfPixelCenters and getIPUVersion() in ['I6E', 'M6', 'I7']:
            mace_check(False, 'Only support `HalfPixelCenters`=False!')
        else:
            HalfPixelCenters_arg = mace_op.arg.add()
            HalfPixelCenters_arg.name = MaceKeyword.mace_HalfPixelCenters_str
            HalfPixelCenters_arg.i = HalfPixelCenters

    def convert_space_batch(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        if op_type == TfliteOpType.BATCH_TO_SPACE_ND.name:
            mace_op.type = MaceOp.BatchToSpaceND.name
        else:
            #SPACE_TO_BATCH_ND
            mace_op.type = MaceOp.SpaceToBatchND.name

    def convert_exp(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'Exp'

    def convert_transpose_conv(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.TransposeConv.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.TransposeConvOptions.TransposeConvOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set padding_type
        padding_arg = mace_op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = opt.Padding()
        #set stride_w
        stride_w_arg = mace_op.arg.add()
        stride_w_arg.name = 'stride_w'
        stride_w_arg.i = opt.StrideW()
        #set stride_h
        stride_h_arg = mace_op.arg.add()
        stride_h_arg.name = 'stride_h'
        stride_h_arg.i = opt.StrideH()
        #set padding_left
        padding_left_arg = mace_op.arg.add()
        padding_left_arg.name = 'padding_left'
        padding_left_arg.i = opt.PaddingLeft()
        #set padding_right
        padding_right_arg = mace_op.arg.add()
        padding_right_arg.name = 'padding_right'
        padding_right_arg.i = opt.PaddingRight()
        #set padding_top
        padding_top_arg = mace_op.arg.add()
        padding_top_arg.name = 'padding_top'
        padding_top_arg.i = opt.PaddingTop()
        #set padding_bottom
        padding_bottom_arg = mace_op.arg.add()
        padding_bottom_arg.name = 'padding_bottom'
        padding_bottom_arg.i = opt.PaddingBottom()

    def convert_slice(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Slice.name
    def convert_stridedslice(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.StridedSlice.name
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.StridedSliceOptions.StridedSliceOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set BeginMask
        BeginMask_arg = mace_op.arg.add()
        BeginMask_arg.name = MaceKeyword.mace_begin_mask_str
        BeginMask_arg.i = opt.BeginMask()
        #set EndMask
        EndMask_arg = mace_op.arg.add()
        EndMask_arg.name = MaceKeyword.mace_end_mask_str
        EndMask_arg.i = opt.EndMask()
        #set EllipsisMask
        EllipsisMask_arg = mace_op.arg.add()
        EllipsisMask_arg.name = MaceKeyword.mace_ellipsis_mask_str
        EllipsisMask_arg.i = opt.EllipsisMask()
        #set NewAxisMask
        NewAxisMask_arg = mace_op.arg.add()
        NewAxisMask_arg.name = MaceKeyword.mace_new_axis_mask_str
        NewAxisMask_arg.i = opt.NewAxisMask()
        #set ShrinkAxisMask
        ShrinkAxisMask_arg = mace_op.arg.add()
        ShrinkAxisMask_arg.name = MaceKeyword.mace_shrink_axis_mask_str
        ShrinkAxisMask_arg.i = opt.ShrinkAxisMask()

    def convert_resize_nearest_neighbor(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.ResizeNearestNeighbor.name
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.ResizeNearestNeighborOptions.ResizeNearestNeighborOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set alignCorners
        align_corners_arg = mace_op.arg.add()
        align_corners_arg.name = MaceKeyword.mace_align_corners_str
        align_corners_arg.i = opt.AlignCorners()
        #set halfPixelCenters
        HalfPixelCenters = opt.HalfPixelCenters()
        if HalfPixelCenters and getIPUVersion() in ['I6E', 'M6', 'I7']:
            mace_check(False, 'Only support `HalfPixelCenters`=False!')
        else:
            halfpixelcenters_arg = mace_op.arg.add()
            halfpixelcenters_arg.name = MaceKeyword.mace_HalfPixelCenters_str
            halfpixelcenters_arg.i = HalfPixelCenters


    def convert_batchmatmul(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.BatchMatmul.name

    def convert_sqrt(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Sqrt.name

    def convert_rsqrt(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Rsqrt.name

    def convert_square(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Square.name

    def convert_abs(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Abs.name

    def convert_tile(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Tile.name

    def convert_pack(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Pack.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.PackOptions.PackOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set axis_value
        values_count = opt.ValuesCount()
        values_count_arg = mace_op.arg.add()
        values_count_arg.name = 'values_count'
        values_count_arg.i = values_count

        axis_value = opt.Axis()
        axis_arg = mace_op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value

    def convert_unpack(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Unpack.name
        #Get option
        op_opt = tflite_op.BuiltinOptions()
        opt = tflite.UnpackOptions.UnpackOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        #set axis_value
        Num = opt.Num()
        Num_arg = mace_op.arg.add()
        Num_arg.name = 'num'
        Num_arg.i = Num

        axis_value = opt.Axis()
        axis_arg = mace_op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value


    def convert_custom(self, tflite_op, op_type, subgraph):
        # customer op not convert, read all flexbuffer options and write back
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        op_code = self._model.OperatorCodes(tflite_op.OpcodeIndex())
        mace_op.name = 'Custom' # for SGSModel_converter_from_Mace: getDataFromFlexBuffer
        mace_op.type = 'Custom'
        custom_opcode = mace_op.arg.add()
        custom_opcode.name = 'custom_opcode'
        custom_opcode.str = str(op_code.CustomCode(), encoding = "utf-8")
        custom_options = mace_op.arg.add()
        custom_options.name = 'custom_options'
        custom_options.s = struct.pack('B' * tflite_op.CustomOptionsLength(),
            *list(tflite_op.CustomOptionsAsNumpy()))

    def convert_shape(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Shape.name

    def convert_less(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'Less'

    def convert_greater(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Greater.name

    def convert_greaterequal(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'GreaterEqual'

    def convert_equal(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'Equal'

    def convert_round(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'Round'

    def convert_gather(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Gather.name

    def convert_topkv2(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'Topkv2'

    def convert_log(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = MaceOp.Log.name

    def convert_sin(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'Sin'

    def convert_cos(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'Cos'

    def convert_floordiv(self, tflite_op, op_type, subgraph):
        mace_op = self.convert_general_op(tflite_op, op_type, subgraph)
        mace_op.type = 'FloorDiv'
