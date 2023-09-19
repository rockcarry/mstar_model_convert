import sys
import os
import struct
import six
import ctypes
from anchor_param import  *
import pdb
if 'IPU_TOOL' in os.environ:
    Project_path = os.environ['IPU_TOOL']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/ConvertTool"))
elif 'SGS_IPU_DIR' in os.environ:
    Project_path = os.environ['SGS_IPU_DIR']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/ConvertTool"))
else:
    raise OSError('Run `source cfg_env.sh` in top directory.')
from third_party import tflite
from third_party.tflite import Model
from third_party.tflite import BuiltinOperator
from third_party.python import flatbuffers

from mace.python.tools.convert_util import getIPUVersion
from mace.python.tools.convert_util import mace_check



class TFLitePostProcess:

    def __init__(self):
        self.builder = flatbuffers.Builder(0)
        self.BuiltinOperator = tflite.BuiltinOperator.BuiltinOperator()
        self.TensorType = tflite.TensorType.TensorType()
        self.CustomOptionsFormat = tflite.CustomOptionsFormat.CustomOptionsFormat()
        if 'IPU_TOOL' in os.environ:
            Project_path = os.environ['IPU_TOOL']
            self.lib = ctypes.cdll.LoadLibrary(os.path.join(Project_path, "libs/x86_64/libSGSCusOP.so"))
        elif 'SGS_IPU_DIR' in os.environ:
            Project_path = os.environ['SGS_IPU_DIR']
            self.lib = ctypes.cdll.LoadLibrary(os.path.join(Project_path, "libs/x86_64/libSGSCusOP.so"))
        else:
            raise OSError('Run `source cfg_env.sh` in top directory.')
        self.null_quant = self.createQuantizationParameters([],[],[],[])

        self.model = None
        self.operator_codes = []
        self.operator_codes_dict = {}
        self.subgraphs = []
        self.tensors =[]
        self.tensors_dict ={}
        self.operators = []
        self.operators_dict = {}
        self.buffers = []
        self.buffers_dict = {}

    def setConfig(self, config):
        self.shape = config["shape"]
        self.tx_func = config["tx_func"]
        self.ty_func = config["ty_func"]
        self.tw_func = config["tw_func"]
        self.th_func = config["th_func"]

        self.x_scale = config["x_scale"]
        self.y_scale = config["y_scale"]
        self.w_scale = config["w_scale"]
        self.h_scale = config["h_scale"]

        self.anchor_selector = config["anchor_selector"]
        self.pw = config["pw"]
        self.ph = config["ph"]
        self.pw_func = config["pw_func"]
        self.ph_func = config["ph_func"]
        self.ppw = config["ppw"]
        self.px = config["px"]
        self.pph = config["pph"]
        self.py = config["py"]
        self.sx = config["sx"]
        self.sy = config["sy"]
        self.sw = config["sw"]
        self.sh = config["sh"]

    def createReshapeOptions(self, new_shape):
        '''

        :param new_shape: a list of int
        :return: offset of the reshape options
        '''
        tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(self.builder, len(new_shape))
        for shape in reversed(new_shape):
            self.builder.PrependInt32(int(shape))
        new_shape_offset = self.builder.EndVector(len(new_shape))
        tflite.ReshapeOptions.ReshapeOptionsStart(self.builder)
        tflite.ReshapeOptions.ReshapeOptionsAddNewShape(self.builder, new_shape_offset)
        reshape_options = tflite.ReshapeOptions.ReshapeOptionsEnd(self.builder)
        return reshape_options

    def createConcatenationOptions(self, axis,fused_activation_function):
        '''

        :param new_shape: a list of int
        :return: offset of the reshape options
        '''
        tflite.ConcatenationOptions.ConcatenationOptionsStart(self.builder)
        tflite.ConcatenationOptions.ConcatenationOptionsAddAxis(self.builder, axis)
        tflite.ConcatenationOptions.ConcatenationOptionsAddFusedActivationFunction(self.builder, fused_activation_function)
        concat_options = tflite.ConcatenationOptions.ConcatenationOptionsEnd(self.builder)
        return concat_options

    def createStridedSliceOptions(self, beginMask,endMask,ellipsisMask,newAxisMask,shrinkAxisMask):
        '''

        :param new_shape: a list of int
        :return: offset of the reshape options
        '''
        tflite.StridedSliceOptions.StridedSliceOptionsStart(self.builder)
        tflite.StridedSliceOptions.StridedSliceOptionsAddBeginMask(self.builder, beginMask)
        tflite.StridedSliceOptions.StridedSliceOptionsAddEndMask(self.builder, endMask)
        tflite.StridedSliceOptions.StridedSliceOptionsAddEllipsisMask(self.builder, ellipsisMask)
        tflite.StridedSliceOptions.StridedSliceOptionsAddNewAxisMask(self.builder, newAxisMask)
        tflite.StridedSliceOptions.StridedSliceOptionsAddShrinkAxisMask(self.builder, shrinkAxisMask)
        stridedslice_options = tflite.StridedSliceOptions.StridedSliceOptionsEnd(self.builder)
        return stridedslice_options

    def createUnpackOptions(self, num, axis):
        '''
        :param new_shape: a list of int
        :return: offset of the reshape options
        '''
        tflite.UnpackOptions.UnpackOptionsStart(self.builder)
        tflite.UnpackOptions.UnpackOptionsAddNum(self.builder, num)
        tflite.UnpackOptions.UnpackOptionsAddAxis(self.builder, axis)
        unpack_options = tflite.UnpackOptions.UnpackOptionsEnd(self.builder)
        return unpack_options

    def createConv2DOptions(self,padding,strideW,strideH,fusedActivationFunction,dilationWFactor,dilationHFactor,paddingLeft,paddingRight,paddingTop,paddingBottom):
        '''
        :param new_shape: a list of int
        :return: offset of the reshape options
        '''
        tflite.Conv2DOptions.Conv2DOptionsStart(self.builder)
        tflite.Conv2DOptions.Conv2DOptionsAddPadding(self.builder, padding)
        tflite.Conv2DOptions.Conv2DOptionsAddStrideW(self.builder, strideW)
        tflite.Conv2DOptions.Conv2DOptionsAddStrideH(self.builder, strideH)
        tflite.Conv2DOptions.Conv2DOptionsAddFusedActivationFunction(self.builder, fusedActivationFunction)
        tflite.Conv2DOptions.Conv2DOptionsAddDilationWFactor(self.builder, dilationWFactor)
        tflite.Conv2DOptions.Conv2DOptionsAddDilationHFactor(self.builder, dilationHFactor)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingLeft(self.builder, paddingLeft)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingRight(self.builder, paddingRight)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingTop(self.builder, paddingTop)
        tflite.Conv2DOptions.Conv2DOptionsAddPaddingBottom(self.builder, paddingBottom)
        Conv2D_options = tflite.Conv2DOptions.Conv2DOptionsEnd(self.builder)
        return Conv2D_options

    def createDepthWiseConv2DOptions(self,padding,strideW,strideH,fusedActivationFunction,dilationWFactor,dilationHFactor,paddingLeft,paddingRight,paddingTop,paddingBottom,depthMultiplier=1):
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsStart(self.builder)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPadding(self.builder, padding)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideW(self.builder, strideW)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideH(self.builder, strideH)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDepthMultiplier(self.builder, depthMultiplier)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddFusedActivationFunction(self.builder, fusedActivationFunction)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationWFactor(self.builder, dilationWFactor)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationHFactor(self.builder, dilationHFactor)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingLeft(self.builder, paddingLeft)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingRight(self.builder, paddingRight)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingTop(self.builder, paddingTop)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingBottom(self.builder, paddingBottom)
        DepthWiseConv2D_options = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsEnd(self.builder)
        return DepthWiseConv2D_options

    def createPool2DOptions(self,padding,strideW,strideH,filterWidth,filterHeight,fusedActivationFunction,paddingLeft,paddingRight,paddingTop,paddingBottom):
        tflite.Pool2DOptions.Pool2DOptionsStart(self.builder)
        tflite.Pool2DOptions.Pool2DOptionsAddPadding(self.builder, padding)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideW(self.builder, strideW)
        tflite.Pool2DOptions.Pool2DOptionsAddStrideH(self.builder, strideH)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(self.builder, filterWidth)
        tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(self.builder, filterHeight)
        tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(self.builder, fusedActivationFunction)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingLeft(self.builder, paddingLeft)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingRight(self.builder, paddingRight)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingTop(self.builder, paddingTop)
        tflite.Pool2DOptions.Pool2DOptionsAddPaddingBottom(self.builder, paddingBottom)
        Pool2D_options = tflite.Pool2DOptions.Pool2DOptionsEnd(self.builder)
        return Pool2D_options

    def creatSplitOptions(self,output_num):
        '''
        :param output_num: output num
        :return: offset of the Split options
        '''
        tflite.SplitOptions.SplitOptionsStart(self.builder)
        tflite.SplitOptions.SplitOptionsAddNumSplits(self.builder, output_num)
        Split_optons = tflite.SplitOptions.SplitOptionsEnd(self.builder)
        return Split_optons

    def createFlexBuffer(self, values):
        '''
        values: an array of tuples(string,value[int/float],string["int"/"float"])

        return an encoded flexbuffer bytearray
        '''

        self.lib.startCreatFlexBuffer.restype = None

        self.lib.startCreatFlexBuffer()
        for value in values:
            if value[-1] == "int":
                self.lib.insertIntString.argtypes = [ctypes.c_char_p,ctypes.c_int]
                self.lib.insertIntString.restype = None
                value_name = value[0]
                value_name = bytes(value_name)
                self.lib.insertIntString(value_name,value[1])
            elif value[-1] == "float":
                self.lib.insertFloatString.argtypes = [ctypes.c_char_p,ctypes.c_float]
                self.lib.insertFloatString.restype = None
                value_name = value[0]
                value_name = bytes(value_name)
                self.lib.insertFloatString(value_name,value[1])
            else:
                print('\033[31mOnly Int and Float supported.\033[0m')

        c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
        self.lib.getFlexBufferData.restype = c_ubyte_p
        cusData = self.lib.getFlexBufferData()

        self.lib.getFlexBufferLenth.restype = ctypes.c_int
        bufferLen = self.lib.getFlexBufferLenth()

        _allByteArray = bytearray()
        for i in six.moves.range(bufferLen):
           _allByteArray.append(cusData[i])

        self.lib.endCreatFlexBuffer.restype = None
        self.lib.endCreatFlexBuffer()
        return _allByteArray

    def createModel(self, version, operator_codes, subgraphs, description,buffers,metadata_buffer=None):
        '''
        // Version of the schema.
        version:uint;

        // A list of all operator codes used in this model. This is
        // kept in order because operators carry an index into this
        // vector.
        operator_codes:[OperatorCode];

        // All the subgraphs of the model. The 0th is assumed to be the main
        // model.
        subgraphs:[SubGraph];

        // A description of the model.
        description:string;

        // Buffers of the model.
        // Note the 0th entry of this array must be an empty buffer (sentinel).
        // This is a convention so that tensors without a buffer can provide 0 as
        // their buffer.
        buffers:[Buffer];

        Metadata about the model.  Indirects into the existings buffers list.
          metadata_buffer:[int];
        :return:
        '''
        tflite.Model.ModelStartOperatorCodesVector(self.builder, len(operator_codes))
        for op_code in reversed(operator_codes):
          self.builder.PrependUOffsetTRelative(int(op_code))
        operator_codes = self.builder.EndVector(len(operator_codes))

        tflite.Model.ModelStartBuffersVector(self.builder, len(buffers))
        for buffer in reversed(buffers):
          self.builder.PrependUOffsetTRelative(int(buffer))
        buffers = self.builder.EndVector(len(buffers))

        tflite.Model.ModelStartOperatorCodesVector(self.builder, len(subgraphs))
        for subgraph in subgraphs:
          self.builder.PrependUOffsetTRelative(int(subgraph))
        subgraphs = self.builder.EndVector(len(subgraphs))
        description = self.builder.CreateString('Sigmastar Postprocess')
        tflite.Model.ModelStart(self.builder)
        tflite.Model.ModelAddVersion(self.builder, version)
        tflite.Model.ModelAddDescription(self.builder, description)
        tflite.Model.ModelAddOperatorCodes(self.builder,operator_codes)
        tflite.Model.ModelAddSubgraphs(self.builder, subgraphs)
        tflite.Model.ModelAddBuffers(self.builder, buffers)
        model = tflite.Model.ModelEnd(self.builder)

        return model

    def createOperatorCode(self,builtin_code, custom_code, version=1):
        '''
        builtin_code:BuiltinOperator;
        custom_code:string;

        // The version of the operator. The version need to be bumped whenever new
        // parameters are introduced into an op.
        version:int = 1;
        :return:
        '''
        custom_code = self.builder.CreateString(custom_code)
        tflite.OperatorCode.OperatorCodeStart(self.builder)
        if getIPUVersion() == 'M6' or getIPUVersion() == 'I6E':
            if builtin_code > 127:#schema 1.14 The maximum number of operators supported is 128(0~127)
                mace_check(0, 'Model schema version is higher than 1.14, Please check model !! \n ')
            tflite.OperatorCode.OperatorCodeAddDeprecatedBuiltinCode(self.builder, builtin_code)
        else:
            tflite.OperatorCode.OperatorCodeAddBuiltinCode(self.builder, builtin_code)
        tflite.OperatorCode.OperatorCodeAddCustomCode(self.builder, custom_code)
        tflite.OperatorCode.OperatorCodeAddVersion(self.builder, version)
        operator_code = tflite.OperatorCode.OperatorCodeEnd(self.builder)
        return operator_code

    def createQuantizationParameters(self,min,max,scale,zero_point,quantized_dimension=None,details=None):
        """
          // These four parameters are the asymmetric linear quantization parameters.
          // Given a quantized value q, the corresponding float value f should be:
          //   f = scale * (q - zero_point)
          // For other quantization types, the QuantizationDetails below is used.
          min:[float];  // For importing back into tensorflow.
          max:[float];  // For importing back into tensorflow.
          scale:[float];  // For dequantizing the tensor's values.
          zero_point:[long];

          // If this is not none, the other quantization parameters (i.e. min, max,
          // scale, zero_point fields above) are ignored and the value of the
          // QuantizationDetails union should be used.
          details:QuantizationDetails;

          // Specifies the dimension of the Tensor's shape that the scales and
          // zero_points correspond to. For example, a tensor t, with dims=[4, 3, 2, 1]
          // with quantization params:
          //   scale=[1.0, 2.0, 3.0], zero_point=[1, 2, 3], quantization_dimension=1
          // will be quantized across the second dimension of t.
          //   t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
          //   t[:, 1, :, :] will have scale[1]=2.0, zero_point[0]=2
          //   t[:, 2, :, :] will have scale[2]=3.0, zero_point[0]=3
          quantized_dimension:int;
        """
        tflite.QuantizationParameters.QuantizationParametersStartMinVector(self.builder, len(min))
        for m in reversed(min):
          self.builder.PrependFloat64(m)
        min_offset = self.builder.EndVector(len(min))

        tflite.QuantizationParameters.QuantizationParametersStartMaxVector(self.builder, len(max))
        for m in reversed(max):
          self.builder.PrependFloat64(m)
        max_offset = self.builder.EndVector(len(max))

        tflite.QuantizationParameters.QuantizationParametersStartScaleVector(self.builder, len(scale))
        for s in reversed(scale):
          self.builder.PrependFloat64(s)
        scale_offset = self.builder.EndVector(len(scale))

        tflite.QuantizationParameters.QuantizationParametersStartZeroPointVector(self.builder, len(zero_point))
        for z in reversed(zero_point):
          self.builder.PrependFloat64(z)
        zero_point_offset = self.builder.EndVector(len(zero_point))

        tflite.QuantizationParameters.QuantizationParametersStart(self.builder)
        tflite.QuantizationParameters.QuantizationParametersAddMin(self.builder,min_offset)
        tflite.QuantizationParameters.QuantizationParametersAddMax(self.builder,max_offset)
        tflite.QuantizationParameters.QuantizationParametersAddScale(self.builder,scale_offset)
        tflite.QuantizationParameters.QuantizationParametersAddZeroPoint(self.builder,zero_point_offset)
        quantized_parameter = tflite.QuantizationParameters.QuantizationParametersEnd(self.builder)
        return quantized_parameter

    def createTensor(self,shape,type,buffer,name,is_variable,quantization=None):
        '''
        shape:[int];
        type:TensorType;
        buffer:uint;
        name:string;
        quantization:QuantizationParameters;
        is_variable:bool = false;
        '''
        tensor_name = self.builder.CreateString(name)
        tflite.Tensor.TensorStartShapeVector(self.builder,len(shape))
        for i in reversed(shape):
            self.builder.PrependInt32(i)
        shapes = self.builder.EndVector(len(shape))
        tflite.Tensor.TensorStart(self.builder)
        tflite.Tensor.TensorAddShape(self.builder,shapes)
        tflite.Tensor.TensorAddType(self.builder,type)
        tflite.Tensor.TensorAddBuffer(self.builder,buffer)

        if quantization is not None:
            tflite.Tensor.TensorAddQuantization(self.builder,quantization)

        tflite.Tensor.TensorAddName(self.builder,tensor_name)
        tflite.Tensor.TensorAddIsVariable(self.builder,is_variable)
        tensor = tflite.Tensor.TensorEnd(self.builder)
        return tensor

    def createOperator(self,opcode_index,inputs,outputs,builtin_options_type=None,builtin_options = None,
                       custom_options = None, mutating_variable_inputs=[]):
        '''
        // Index into the operator_codes array. Using an integer here avoids
         // complicate map lookups.
        opcode_index:uint;

        // Optional input and output tensors are indicated by -1.
        inputs:[int];
        outputs:[int];

        builtin_options:BuiltinOptions;
        custom_options:[ubyte];
        custom_options_format:CustomOptionsFormat;

        // A list of booleans indicating the input tensors which are being mutated by
        // this operator.(e.g. used by RNN and LSTM).
        // For example, if the "inputs" array refers to 5 tensors and the second and
        // fifth are mutable variables, then this list will contain
        // [false, true, false, false, true].
        //
        // If the list is empty, no variable is mutated in this operator.
        // The list either has the same length as `inputs`, or is empty.
        mutating_variable_inputs:[bool];
        '''

        data = []
        tflite.Operator.OperatorStartInputsVector(self.builder,len(inputs))
        for input in reversed(inputs):
            self.builder.PrependInt32(input)
        inputs = self.builder.EndVector(len(inputs))

        tflite.Operator.OperatorStartOutputsVector(self.builder,len(outputs))
        for output in reversed(outputs):
            self.builder.PrependInt32(output)
        outputs =self.builder.EndVector(len(outputs))

        if custom_options is not None:
            tflite.Operator.OperatorStartCustomOptionsVector(self.builder,len(custom_options))
            print(custom_options)
            for meta_ubyte in reversed(custom_options):
                self.builder.PrependByte(meta_ubyte)
            data = self.builder.EndVector(len(custom_options))
        else :
            data = None

        tflite.Operator.OperatorStart(self.builder)
        tflite.Operator.OperatorAddOpcodeIndex(self.builder,opcode_index)
        tflite.Operator.OperatorAddInputs(self.builder,inputs)
        tflite.Operator.OperatorAddOutputs(self.builder,outputs)
        if builtin_options is not None:
            tflite.Operator.OperatorAddBuiltinOptionsType(self.builder,builtin_options_type)
            tflite.Operator.OperatorAddBuiltinOptions(self.builder,builtin_options)
        if data is not None:
            tflite.Operator.OperatorAddCustomOptions(self.builder,data)
        operator = tflite.Operator.OperatorEnd(self.builder)
        return operator

    def createSubGraph(self,tensors,inputs,outputs,operators,name):
        """
        // A list of all tensors used in this subgraph.
        tensors:[Tensor];

        // Indices of the tensors that are inputs into this subgraph. Note this is
        // the list of non-static tensors that feed into the subgraph for inference.
        inputs:[int];

        // Indices of the tensors that are outputs out of this subgraph. Note this is
        // the list of output tensors that are considered the product of the
        // subgraph's inference.
        outputs:[int];

        // All operators, in execution order.
        operators:[Operator];

        // Name of this subgraph (used for debugging).
        name:string;
        :return:
        """

        tflite.SubGraph.SubGraphStartTensorsVector(self.builder, len(tensors))
        for tensor in reversed(tensors):
          self.builder.PrependUOffsetTRelative(int(tensor))
        tensors = self.builder.EndVector(len(tensors))

        tflite.SubGraph.SubGraphStartOperatorsVector(self.builder,len(operators))
        for operator in reversed(operators) :
            self.builder.PrependUOffsetTRelative(int(operator))
        operators = self.builder.EndVector(len(operators))

        tflite.SubGraph.SubGraphStartInputsVector(self.builder,len(inputs))
        for input in reversed (inputs):
            self.builder.PrependInt32(input)
        inputs = self.builder.EndVector(len(inputs))

        tflite.SubGraph.SubGraphStartOutputsVector(self.builder,len(outputs))
        for output in reversed(outputs):
          self.builder.PrependInt32(output)
        outputs = self.builder.EndVector(len(outputs))
        name = self.builder.CreateString(name)
        tflite.SubGraph.SubGraphStart(self.builder)
        tflite.SubGraph.SubGraphAddTensors(self.builder, tensors)
        tflite.SubGraph.SubGraphAddInputs(self.builder,inputs)
        tflite.SubGraph.SubGraphAddOutputs(self.builder,outputs)
        tflite.SubGraph.SubGraphAddOperators(self.builder,operators)
        tflite.SubGraph.SubGraphAddName(self.builder,name)
        subgraph = tflite.SubGraph.SubGraphEnd(self.builder)
        return subgraph

    def createBuffer(self, databytes= None):
        """
        data:[ubyte];// (force_align: 16);
        :param data:
        :return:
        """
        data = None
        if databytes is None:
            tflite.Buffer.BufferStartDataVector(self.builder,0)
            data = self.builder.EndVector(0)
        else:
            tflite.Buffer.BufferStartDataVector(self.builder,len(databytes))
            for i in reversed(databytes):
                self.builder.PrependByte(i)
            data = self.builder.EndVector(len(databytes))

        tflite.Buffer.BufferStart(self.builder)
        tflite.Buffer.BufferAddData(self.builder,data)
        buffer = tflite.Buffer.BufferEnd(self.builder)
        return buffer


    def buildBuffer(self, buffer_name, buffer_data=None):
        """
        create buffer with a specific name
        :param buffer_name:
        :param buffer_data: None:a buffer for variable tensor
        :return:
        """

        tmp_buffer = self.createBuffer(buffer_data)
        self.buffers_dict[buffer_name] = (buffer_data,len(self.buffers))
        self.buffers.append(tmp_buffer)
        return self.buffers_dict[buffer_name][-1]

    def getBufferByName(self,buffer_name):
        """

        :param buffer_name:
        :return:
        """
        if buffer_name in self.buffers_dict:
            return self.buffers_dict[buffer_name][-1]
        else:
            return None

    def buildOperatorCode(self, opcode_name, builtin_code, custom_code=None):
        """

        :param opcode_name:
        :param builtin_code:
        :param custom_code:
        :return:
        """
        if self.operator_codes_dict.__contains__(opcode_name):
            return self.operator_codes_dict[opcode_name][-1]
        elif builtin_code == self.BuiltinOperator.CUSTOM:
            tmp_opcode = self.createOperatorCode(builtin_code,custom_code)
            self.operator_codes_dict[opcode_name] = (builtin_code,custom_code,len(self.operator_codes))
            self.operator_codes.append(tmp_opcode)
            return self.operator_codes_dict[opcode_name][-1]
        else:
            cus_code = b'Builtin'
            tmp_opcode = self.createOperatorCode(builtin_code,cus_code)
            self.operator_codes_dict[opcode_name] = (builtin_code,custom_code,len(self.operator_codes))
            self.operator_codes.append(tmp_opcode)
            return self.operator_codes_dict[opcode_name][-1]

    def getOperatorCodeByName(self,opcode_name):
        """

        :param opcode_name:
        :return:
        """
        if self.operator_codes_dict.__contains__(opcode_name):
            return self.operator_codes_dict[opcode_name][-1]
        else:
            return None

    def buildTensor(self, shape, name, buffer=-1,type = tflite.TensorType.TensorType().FLOAT32):
        """
        :param shape:
        :param name:
        :param buffer:buffer id in top
        :return:
        """
        if self.tensors_dict.__contains__(name):
            return self.tensors_dict[name][-1]
        else:
            if buffer == -1:
                buffer_id = self.getBufferByName(name)
                if buffer_id == None:
                    self.buildBuffer(name)
                    buffer_id = self.getBufferByName(name)
            else:
                buffer_id = buffer
            tmp_tensor = self.createTensor(shape,type,buffer_id,name,False,self.null_quant)
            self.tensors_dict[name] = (shape,buffer_id,len(self.tensors))
            self.tensors.append(tmp_tensor)
        return self.tensors_dict[name][-1]

    def getTensorByName(self, name):
        """

        :param name:
        :return:
        """
        if self.tensors_dict.__contains__(name):
            return self.tensors_dict[name][-1]
        else:
            return None

    def buildOperator(self, op_code_name, input_names, output_names,builtin_options_type = None,builtin_options=None, custom_options = None, is_custom=False):
        """

        :param op_code_name:
        :param input_names:
        :param output_names:
        :param custom_data:
        :param is_custom:
        :return:
        """
        opcode = self.getOperatorCodeByName(op_code_name)
        if opcode is None:
            return None

        inputs = []
        for input_name in input_names:
            input = self.getTensorByName(input_name)
            assert input is not None
            inputs.append(input)

        outputs = []
        for output_name in output_names:
            output = self.getTensorByName(output_name)
            assert output is not None
            outputs.append(output)

        id = len(self.operators)
        tmp_operator = self.createOperator(opcode,inputs,outputs,builtin_options_type,builtin_options,custom_options)
        self.operators_dict[id] = (op_code_name,id)
        self.operators.append(tmp_operator)

        return id

    def buildSubGraph(self,input_tensor_names,output_tensor_names,subgraph_name):
        """

        :param input_tensor_names:
        :param output_tensor_names:
        :param subgraph_name:
        :return:
        """
        input_tensors = []
        for name in input_tensor_names:
            input_tensors.append(self.getTensorByName(name))
        output_tensors = []
        for name in output_tensor_names:
            output_tensors.append(self.getTensorByName(name))
        subgraph = self.createSubGraph(self.tensors,input_tensors,output_tensors,self.operators,subgraph_name)
        return subgraph

    def buildBoxDecoding(self,unpacked_box):
        """

        :unpacked_box a list of tensors which are unpacked:
        :return:a list of tensors decoded
        """

        tx_out_tensors = []
        tx_in_tensors = []
        tx_in_tensors.append(unpacked_box[0])
        if self.tx_func[1] is None:
            self.buildTensor(self.shape,"tx_tensor")
        elif self.tx_func[1] == "x_scale":
            self.buildTensor(self.shape,"tx_tensor")
            x_scale = bytearray(struct.pack("f", self.x_scale))
            self.buildBuffer("x_scale",x_scale)
            self.buildTensor([1],"x_scale",self.getBufferByName("x_scale"))
            tx_in_tensors.append("x_scale")
        else:
            None

        tx_out_tensors.append("tx_tensor")
        self.buildOperatorCode("SGS_tx",self.tx_func[0])
        self.buildOperator("SGS_tx",tx_in_tensors,tx_out_tensors)

        ty_out_tensors = []
        ty_in_tensors = []
        ty_in_tensors.append(unpacked_box[1])
        if self.ty_func[1] is None:
            self.buildTensor(self.shape,"ty_tensor")
        elif self.ty_func[1] == "y_scale":
            self.buildTensor(self.shape,"ty_tensor")
            y_scale = bytearray(struct.pack("f", self.y_scale))
            self.buildBuffer("y_scale",y_scale)
            self.buildTensor([1],"y_scale",self.getBufferByName("y_scale"))
            ty_in_tensors.append("y_scale")
        else:
            None

        ty_out_tensors.append("ty_tensor")
        self.buildOperatorCode("SGS_ty",self.ty_func[0])
        self.buildOperator("SGS_ty",ty_in_tensors,ty_out_tensors)

        tw_out_tensors = []
        tw_in_tensors = []
        if(self.tw_func[0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
            tw_in_tensors.append(unpacked_box[2])
            reshape_vector=[]
            for value in self.shape:
                reshape_vector += bytearray(struct.pack("i", value))
            self.buildBuffer("id_reshape_vector",reshape_vector)
            self.buildTensor([len(self.shape)],"id_reshape_shape",self.getBufferByName("id_reshape_vector"),tflite.TensorType.TensorType().INT32)
            tw_in_tensors.append("id_reshape_shape")
            self.buildTensor(self.shape,"tw_tensor")
            tw_out_tensors.append("tw_tensor")
            self.buildOperatorCode("SGS_tw",self.tw_func[0])
            id_reshape_newshape = self.createReshapeOptions(self.shape)
            self.buildOperator("SGS_tw",tw_in_tensors,tw_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,id_reshape_newshape)

        else:
            tw_in_tensors.append(unpacked_box[2])
            if self.tw_func[1] is None:
                self.buildTensor(self.shape,"tw_tensor")
            elif self.tw_func[1] == "w_scale":
                self.buildTensor(self.shape,"tw_tensor")
                w_scale = bytearray(struct.pack("f", self.w_scale))
                self.buildBuffer("w_scale",w_scale)
                self.buildTensor([1],"w_scale",self.getBufferByName("w_scale"))
                tw_in_tensors.append("w_scale")
            else:
                None

            tw_out_tensors.append("tw_tensor")
            self.buildOperatorCode("SGS_tw",self.tw_func[0])
            self.buildOperator("SGS_tw",tw_in_tensors,tw_out_tensors)


        th_out_tensors = []
        th_in_tensors = []
        if(self.th_func[0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
            th_in_tensors.append(unpacked_box[3])
            reshape_vector=[]
            for value in self.shape:
                reshape_vector += bytearray(struct.pack("i", value))
            self.buildBuffer("id_reshape_vector",reshape_vector)
            self.buildTensor([len(self.shape)],"id_reshape_shape",self.getBufferByName("id_reshape_vector"),tflite.TensorType.TensorType().INT32)
            th_in_tensors.append("id_reshape_shape")
            self.buildTensor(self.shape,"th_tensor")
            th_out_tensors.append("th_tensor")
            self.buildOperatorCode("SGS_th",self.th_func[0])
            id_reshape_newshape = self.createReshapeOptions(self.shape)
            self.buildOperator("SGS_th",th_in_tensors,th_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,id_reshape_newshape)
        else:
            th_in_tensors.append(unpacked_box[3])
            if self.th_func[1] is None:
                self.buildTensor(self.shape,"th_tensor")
            elif self.th_func[1] == "h_scale":
                self.buildTensor(self.shape,"th_tensor")
                h_scale = bytearray(struct.pack("f", self.h_scale))
                self.buildBuffer("h_scale",h_scale)
                self.buildTensor([1],"h_scale",self.getBufferByName("h_scale"))
                th_in_tensors.append("h_scale")
            else:
                None

            th_out_tensors.append("th_tensor")
            self.buildOperatorCode("SGS_th",self.th_func[0])
            self.buildOperator("SGS_th",th_in_tensors,th_out_tensors)


        if self.anchor_selector == "constant":
            ph_vector=[]
            for value in self.ph:
                ph_vector += bytearray(struct.pack("f", value))
            self.buildBuffer("ph_buffer",ph_vector)
            self.buildTensor([len(self.ph)],"ph_tensor",self.getBufferByName("ph_buffer"))

            pw_vector=[]
            for value in self.pw:
                pw_vector += bytearray(struct.pack("f", value))
            self.buildBuffer("pw_buffer",pw_vector)
            self.buildTensor([len(self.pw)],"pw_tensor",self.getBufferByName("pw_buffer"))
        else:
            ph_out_tensors = []
            ph_in_tensors = []
            ph_in_tensors.append(unpacked_box[3])
            self.buildTensor(self.shape,"ph_tensor")
            ph_out_tensors.append("ph_tensor")
            self.buildOperatorCode("SGS_ph",self.ph_func[0])
            self.buildOperator("SGS_ph",ph_in_tensors,ph_out_tensors)
            pw_out_tensors = []
            pw_in_tensors = []
            pw_in_tensors.append(unpack_output_tensors[2])
            self.buildTensor(self.shape,"pw_tensor")
            pw_out_tensors.append("pw_tensor")
            self.buildOperatorCode("SGS_pw",self.pw_func[0])
            self.buildOperator("SGS_pw",pw_in_tensors,pw_out_tensors)

        ppw_vector=[]
        for value in self.ppw:
            ppw_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("ppw_buffer",ppw_vector)
        self.buildTensor([len(self.ppw)],"ppw_tensor",self.getBufferByName("ppw_buffer"))

        px_vector=[]
        for value in self.px:
            px_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("px_buffer",px_vector)
        self.buildTensor([len(self.px)],"px_tensor",self.getBufferByName("px_buffer"))

        pph_vector=[]
        for value in self.pph:
            pph_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("pph_buffer",pph_vector)
        self.buildTensor([len(self.pph)],"pph_tensor",self.getBufferByName("pph_buffer"))

        py_vector=[]
        for value in self.py:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"py_tensor",self.getBufferByName("py_buffer"))


        x_multi0_out_tensors = []
        x_multi0_in_tensors = []
        x_multi0_in_tensors.append("tx_tensor")
        x_multi0_in_tensors.append("ppw_tensor")
        self.buildTensor(self.shape,"x_multi0_tensor")
        x_multi0_out_tensors.append("x_multi0_tensor")
        self.buildOperatorCode("SGS_x_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_x_multi0",x_multi0_in_tensors,x_multi0_out_tensors)

        x_add_out_tensors = []
        x_add_in_tensors = []
        x_add_in_tensors.append("x_multi0_tensor")
        x_add_in_tensors.append("px_tensor")
        self.buildTensor(self.shape,"x_add_tensor")
        x_add_out_tensors.append("x_add_tensor")
        self.buildOperatorCode("SGS_x_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
        self.buildOperator("SGS_x_add",x_add_in_tensors,x_add_out_tensors)

        py_vector=[]
        for value in self.sx:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"sx_tensor",self.getBufferByName("py_buffer"))

        py_vector=[]
        for value in self.sy:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"sy_tensor",self.getBufferByName("py_buffer"))

        py_vector=[]
        for value in self.sw:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"sw_tensor",self.getBufferByName("py_buffer"))

        py_vector=[]
        for value in self.sh:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"sh_tensor",self.getBufferByName("py_buffer"))

        x_multi1_out_tensors = []
        x_multi1_in_tensors = []
        x_multi1_in_tensors.append("x_add_tensor")
        x_multi1_in_tensors.append("sx_tensor")
        self.buildTensor(self.shape,"x_multi1_tensor")
        x_multi1_out_tensors.append("x_multi1_tensor")
        self.buildOperatorCode("SGS_x_multi1",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_x_multi1",x_multi1_in_tensors,x_multi1_out_tensors)

        y_multi0_out_tensors = []
        y_multi0_in_tensors = []
        y_multi0_in_tensors.append("ty_tensor")
        y_multi0_in_tensors.append("pph_tensor")
        self.buildTensor(self.shape,"y_multi0_tensor")
        y_multi0_out_tensors.append("y_multi0_tensor")
        self.buildOperatorCode("SGS_y_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_y_multi0",y_multi0_in_tensors,y_multi0_out_tensors)

        y_add_out_tensors = []
        y_add_in_tensors = []
        y_add_in_tensors.append("y_multi0_tensor")
        y_add_in_tensors.append("py_tensor")
        self.buildTensor(self.shape,"y_add_tensor")
        y_add_out_tensors.append("y_add_tensor")
        self.buildOperatorCode("SGS_y_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
        self.buildOperator("SGS_y_add",y_add_in_tensors,y_add_out_tensors)

        y_multi1_out_tensors = []
        y_multi1_in_tensors = []
        y_multi1_in_tensors.append("y_add_tensor")
        y_multi1_in_tensors.append("sy_tensor")
        self.buildTensor(self.shape,"y_multi1_tensor")
        y_multi1_out_tensors.append("y_multi1_tensor")
        self.buildOperatorCode("SGS_y_multi1",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_y_multi1",y_multi1_in_tensors,y_multi1_out_tensors)

        w_exp_out_tensors = []
        w_exp_in_tensors = []
        w_exp_in_tensors.append("tw_tensor")
        self.buildTensor(self.shape,"w_exp_tensor")
        w_exp_out_tensors.append("w_exp_tensor")
        self.buildOperatorCode("SGS_w_exp",tflite.BuiltinOperator.BuiltinOperator().EXP)
        self.buildOperator("SGS_w_exp",w_exp_in_tensors,w_exp_out_tensors)

        w_multi0_out_tensors = []
        w_multi0_in_tensors = []
        w_multi0_in_tensors.append("w_exp_tensor")
        w_multi0_in_tensors.append("pw_tensor")
        self.buildTensor(self.shape,"w_multi0_tensor")
        w_multi0_out_tensors.append("w_multi0_tensor")
        self.buildOperatorCode("SGS_w_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_w_multi0",w_multi0_in_tensors,w_multi0_out_tensors)

        w_multi1_out_tensors = []
        w_multi1_in_tensors = []
        w_multi1_in_tensors.append("w_multi0_tensor")
        w_multi1_in_tensors.append("sw_tensor")
        self.buildTensor(self.shape,"w_multi1_tensor")
        w_multi1_out_tensors.append("w_multi1_tensor")
        self.buildOperatorCode("SGS_w_multi1",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_w_multi1",w_multi1_in_tensors,w_multi1_out_tensors)

        h_exp_out_tensors = []
        h_exp_in_tensors = []
        h_exp_in_tensors.append("th_tensor")
        self.buildTensor(self.shape,"h_exp_tensor")
        h_exp_out_tensors.append("h_exp_tensor")
        self.buildOperatorCode("SGS_h_exp",tflite.BuiltinOperator.BuiltinOperator().EXP)
        self.buildOperator("SGS_h_exp",h_exp_in_tensors,h_exp_out_tensors)

        h_multi0_out_tensors = []
        h_multi0_in_tensors = []
        h_multi0_in_tensors.append("h_exp_tensor")
        h_multi0_in_tensors.append("ph_tensor")
        self.buildTensor(self.shape,"h_multi0_tensor")
        h_multi0_out_tensors.append("h_multi0_tensor")
        self.buildOperatorCode("SGS_h_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_h_multi0",h_multi0_in_tensors,h_multi0_out_tensors)

        h_multi1_out_tensors = []
        h_multi1_in_tensors = []
        h_multi1_in_tensors.append("h_multi0_tensor")
        h_multi1_in_tensors.append("sh_tensor")
        self.buildTensor(self.shape,"h_multi1_tensor")
        h_multi1_out_tensors.append("h_multi1_tensor")
        self.buildOperatorCode("SGS_h_multi1",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_h_multi1",h_multi1_in_tensors,h_multi1_out_tensors)

        x1_out_tensors = []
        x1_in_tensors = []
        x1_in_tensors.append("x_multi1_tensor")
        x1_in_tensors.append("w_multi1_tensor")
        self.buildTensor(self.shape,"x1_tensor")
        x1_out_tensors.append("x1_tensor")
        self.buildOperatorCode("SGS_x1_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
        self.buildOperator("SGS_x1_sub",x1_in_tensors,x1_out_tensors)

        y1_out_tensors = []
        y1_in_tensors = []
        y1_in_tensors.append("y_multi1_tensor")
        y1_in_tensors.append("h_multi1_tensor")
        self.buildTensor(self.shape,"y1_tensor")
        y1_out_tensors.append("y1_tensor")
        self.buildOperatorCode("SGS_y1_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
        self.buildOperator("SGS_y1_sub",y1_in_tensors,y1_out_tensors)

        x2_out_tensors = []
        x2_in_tensors = []
        x2_in_tensors.append("x_multi1_tensor")
        x2_in_tensors.append("w_multi1_tensor")
        self.buildTensor(self.shape,"x2_tensor")
        x2_out_tensors.append("x2_tensor")
        self.buildOperatorCode("SGS_x2_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
        self.buildOperator("SGS_x2_add",x2_in_tensors,x2_out_tensors)

        y2_out_tensors = []
        y2_in_tensors = []
        y2_in_tensors.append("y_multi1_tensor")
        y2_in_tensors.append("h_multi1_tensor")
        self.buildTensor(self.shape,"y2_tensor")
        y2_out_tensors.append("y2_tensor")
        self.buildOperatorCode("SGS_y2_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
        self.buildOperator("SGS_y2_add",y2_in_tensors,y2_out_tensors)
        return x1_out_tensors + y1_out_tensors + x2_out_tensors + y2_out_tensors


    def buildBoxDecoding2(self,unpacked_box):
        """

        :unpacked_box a list of tensors which are unpacked:
        :return:a list of tensors decoded
        """
        tx_out_tensors = []
        tx_in_tensors = []
        tx_in_tensors.append(unpacked_box[0])
        if(self.tx_func[0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
            reshape_vector=[]
            for value in self.shape:
                reshape_vector += bytearray(struct.pack("i", value))
            self.buildBuffer("id_reshape_vector",reshape_vector)
            self.buildTensor([len(self.shape)],"id_reshape_shape",self.getBufferByName("id_reshape_vector"),tflite.TensorType.TensorType().INT32)
            tx_in_tensors.append("id_reshape_shape")
        if self.tx_func[1] is None:
            self.buildTensor(self.shape,"tx_tensor")
        elif self.tx_func[1] == "x_scale":
            self.buildTensor(self.shape,"tx_tensor")
            x_scale = bytearray(struct.pack("f", self.x_scale))
            self.buildBuffer("x_scale",x_scale)
            self.buildTensor([1],"x_scale",self.getBufferByName("x_scale"))
            tx_in_tensors.append("x_scale")
        else:
            None

        tx_out_tensors.append("tx_tensor")
        self.buildOperatorCode("SGS_tx",self.tx_func[0])
        self.buildOperator("SGS_tx",tx_in_tensors,tx_out_tensors)



        ty_out_tensors = []
        ty_in_tensors = []
        ty_in_tensors.append(unpacked_box[1])
        if(self.ty_func[0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
            reshape_vector=[]
            for value in self.shape:
                reshape_vector += bytearray(struct.pack("i", value))
            self.buildBuffer("id_reshape_vector",reshape_vector)
            self.buildTensor([len(self.shape)],"id_reshape_shape",self.getBufferByName("id_reshape_vector"),tflite.TensorType.TensorType().INT32)
            ty_in_tensors.append("id_reshape_shape")
        if self.ty_func[1] is None:
            self.buildTensor(self.shape,"ty_tensor")
        elif self.ty_func[1] == "y_scale":
            self.buildTensor(self.shape,"ty_tensor")
            y_scale = bytearray(struct.pack("f", self.y_scale))
            self.buildBuffer("y_scale",y_scale)
            self.buildTensor([1],"y_scale",self.getBufferByName("y_scale"))
            ty_in_tensors.append("y_scale")
        else:
            None

        ty_out_tensors.append("ty_tensor")
        self.buildOperatorCode("SGS_ty",self.ty_func[0])
        self.buildOperator("SGS_ty",ty_in_tensors,ty_out_tensors)

        tw_out_tensors = []
        tw_in_tensors = []
        if(self.tw_func[0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
            tw_in_tensors.append(unpacked_box[2])
            reshape_vector=[]
            for value in self.shape:
                reshape_vector += bytearray(struct.pack("i", value))
            self.buildBuffer("id_reshape_vector",reshape_vector)
            self.buildTensor([len(self.shape)],"id_reshape_shape",self.getBufferByName("id_reshape_vector"),tflite.TensorType.TensorType().INT32)
            tw_in_tensors.append("id_reshape_shape")
            self.buildTensor(self.shape,"tw_tensor")
            tw_out_tensors.append("tw_tensor")
            self.buildOperatorCode("SGS_tw",self.tw_func[0])
            id_reshape_newshape = self.createReshapeOptions(self.shape)
            self.buildOperator("SGS_tw",tw_in_tensors,tw_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,id_reshape_newshape)

        else:
            tw_in_tensors.append(unpacked_box[2])
            if self.tw_func[1] is None:
                self.buildTensor(self.shape,"tw_tensor")
            elif self.tw_func[1] == "w_scale":
                self.buildTensor(self.shape,"tw_tensor")
                w_scale = bytearray(struct.pack("f", self.w_scale))
                self.buildBuffer("w_scale",w_scale)
                self.buildTensor([1],"w_scale",self.getBufferByName("w_scale"))
                tw_in_tensors.append("w_scale")
            else:
                None

            tw_out_tensors.append("tw_tensor")
            self.buildOperatorCode("SGS_tw",self.tw_func[0])
            self.buildOperator("SGS_tw",tw_in_tensors,tw_out_tensors)


        th_out_tensors = []
        th_in_tensors = []
        if(self.th_func[0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
            th_in_tensors.append(unpacked_box[3])
            reshape_vector=[]
            for value in self.shape:
                reshape_vector += bytearray(struct.pack("i", value))
            self.buildBuffer("id_reshape_vector",reshape_vector)
            self.buildTensor([len(self.shape)],"id_reshape_shape",self.getBufferByName("id_reshape_vector"),tflite.TensorType.TensorType().INT32)
            th_in_tensors.append("id_reshape_shape")
            self.buildTensor(self.shape,"th_tensor")
            th_out_tensors.append("th_tensor")
            self.buildOperatorCode("SGS_th",self.th_func[0])
            id_reshape_newshape = self.createReshapeOptions(self.shape)
            self.buildOperator("SGS_th",th_in_tensors,th_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,id_reshape_newshape)
        else:
            th_in_tensors.append(unpacked_box[3])
            if self.th_func[1] is None:
                self.buildTensor(self.shape,"th_tensor")
            elif self.th_func[1] == "h_scale":
                self.buildTensor(self.shape,"th_tensor")
                h_scale = bytearray(struct.pack("f", self.h_scale))
                self.buildBuffer("h_scale",h_scale)
                self.buildTensor([1],"h_scale",self.getBufferByName("h_scale"))
                th_in_tensors.append("h_scale")
            else:
                None

            th_out_tensors.append("th_tensor")
            self.buildOperatorCode("SGS_th",self.th_func[0])
            self.buildOperator("SGS_th",th_in_tensors,th_out_tensors)


        if self.anchor_selector == "constant":
            ph_vector=[]
            for value in self.ph:
                ph_vector += bytearray(struct.pack("f", value))
            self.buildBuffer("ph_buffer",ph_vector)
            self.buildTensor([len(self.ph)],"ph_tensor",self.getBufferByName("ph_buffer"))

            pw_vector=[]
            for value in self.pw:
                pw_vector += bytearray(struct.pack("f", value))
            self.buildBuffer("pw_buffer",pw_vector)
            self.buildTensor([len(self.pw)],"pw_tensor",self.getBufferByName("pw_buffer"))
        else:
            ph_out_tensors = []
            ph_in_tensors = []
            ph_in_tensors.append(unpacked_box[3])
            self.buildTensor(self.shape,"ph_tensor")
            ph_out_tensors.append("ph_tensor")
            self.buildOperatorCode("SGS_ph",self.ph_func[0])
            self.buildOperator("SGS_ph",ph_in_tensors,ph_out_tensors)
            pw_out_tensors = []
            pw_in_tensors = []
            pw_in_tensors.append(unpack_output_tensors[2])
            self.buildTensor(self.shape,"pw_tensor")
            pw_out_tensors.append("pw_tensor")
            self.buildOperatorCode("SGS_pw",self.pw_func[0])
            self.buildOperator("SGS_pw",pw_in_tensors,pw_out_tensors)

        ppw_vector=[]
        for value in self.ppw:
            ppw_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("ppw_buffer",ppw_vector)
        self.buildTensor([len(self.ppw)],"ppw_tensor",self.getBufferByName("ppw_buffer"))
        self.buildTensor([1,len(self.ppw)],"ppw_div_tensor",self.getBufferByName("ppw_buffer"))

        px_vector=[]
        for value in self.px:
            px_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("px_buffer",px_vector)
        self.buildTensor([len(self.px)],"px_tensor",self.getBufferByName("px_buffer"))

        pph_vector=[]
        for value in self.pph:
            pph_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("pph_buffer",pph_vector)
        self.buildTensor([len(self.pph)],"pph_tensor",self.getBufferByName("pph_buffer"))

        py_vector=[]
        for value in self.py:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"py_tensor",self.getBufferByName("py_buffer"))

        positive_one = [1.0]
        positive_one_vector=[]
        positive_one_vector += bytearray(struct.pack("f", positive_one[0]))
        self.buildBuffer("positive_one_buffer", positive_one_vector)
        self.buildTensor([1],"positive_one_tensor",self.getBufferByName("positive_one_buffer"))

        x_multi0_out_tensors = []
        x_multi0_in_tensors = []
        x_multi0_in_tensors.append("tx_tensor")
        x_multi0_in_tensors.append("positive_one_tensor")
        self.buildTensor(self.shape,"x_multi0_tensor")
        x_multi0_out_tensors.append("x_multi0_tensor")
        self.buildOperatorCode("SGS_x_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_x_multi0",x_multi0_in_tensors,x_multi0_out_tensors)

        x_add_out_tensors = []
        x_add_in_tensors = []
        x_add_in_tensors.append("x_multi0_tensor")
        x_add_in_tensors.append("px_tensor")
        self.buildTensor(self.shape,"x_add_tensor")
        x_add_out_tensors.append("x_add_tensor")
        self.buildOperatorCode("SGS_x_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
        self.buildOperator("SGS_x_add",x_add_in_tensors,x_add_out_tensors)

        py_vector=[]
        for value in self.sx:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"sx_tensor",self.getBufferByName("py_buffer"))

        py_vector=[]
        for value in self.sy:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"sy_tensor",self.getBufferByName("py_buffer"))

        py_vector=[]
        for value in self.sw:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"sw_tensor",self.getBufferByName("py_buffer"))

        py_vector=[]
        for value in self.sh:
            py_vector += bytearray(struct.pack("f", value))
        self.buildBuffer("py_buffer",py_vector)
        self.buildTensor([len(self.py)],"sh_tensor",self.getBufferByName("py_buffer"))

        x_multi1_out_tensors = []
        x_multi1_in_tensors = []
        x_multi1_in_tensors.append("x_add_tensor")
        x_multi1_in_tensors.append("sx_tensor")
        self.buildTensor(self.shape,"x_multi1_tensor")
        x_multi1_out_tensors.append("x_multi1_tensor")
        self.buildOperatorCode("SGS_x_multi1",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_x_multi1",x_multi1_in_tensors,x_multi1_out_tensors)

        y_multi0_out_tensors = []
        y_multi0_in_tensors = []
        y_multi0_in_tensors.append("ty_tensor")
        y_multi0_in_tensors.append("positive_one_tensor")
        self.buildTensor(self.shape,"y_multi0_tensor")
        y_multi0_out_tensors.append("y_multi0_tensor")
        self.buildOperatorCode("SGS_y_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_y_multi0",y_multi0_in_tensors,y_multi0_out_tensors)

        y_add_out_tensors = []
        y_add_in_tensors = []
        y_add_in_tensors.append("y_multi0_tensor")
        y_add_in_tensors.append("py_tensor")
        self.buildTensor(self.shape,"y_add_tensor")
        y_add_out_tensors.append("y_add_tensor")
        self.buildOperatorCode("SGS_y_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
        self.buildOperator("SGS_y_add",y_add_in_tensors,y_add_out_tensors)

        y_multi1_out_tensors = []
        y_multi1_in_tensors = []
        y_multi1_in_tensors.append("y_add_tensor")
        y_multi1_in_tensors.append("sy_tensor")
        self.buildTensor(self.shape,"y_multi1_tensor")
        y_multi1_out_tensors.append("y_multi1_tensor")
        self.buildOperatorCode("SGS_y_multi1",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_y_multi1",y_multi1_in_tensors,y_multi1_out_tensors)

        w_sub_out_tensors = []
        w_sub_in_tensors = []
        w_sub_in_tensors.append("tw_tensor")
        w_sub_in_tensors.append("positive_one_tensor")
        self.buildTensor(self.shape,"w_sub_tensor")
        w_sub_out_tensors.append("w_sub_tensor")
        self.buildOperatorCode("SGS_w_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
        self.buildOperator("SGS_w_sub",w_sub_in_tensors,w_sub_out_tensors)

        w_sub_div_out_tensors = []
        w_sub_div_in_tensors = []
        w_sub_div_in_tensors.append("w_sub_tensor")
        self.buildTensor(self.shape,"w_sub_reciprocal_tensor")
        w_sub_div_out_tensors.append("w_sub_reciprocal_tensor")
        cus_code = 'Reciprocal'
        cus_options = [(b"numeratort",1,"int")]
        options = self.createFlexBuffer(cus_options)
        self.buildOperatorCode("SGS_w_sub_reciprocal",self.BuiltinOperator.CUSTOM, cus_code)
        self.buildOperator("SGS_w_sub_reciprocal",w_sub_div_in_tensors,w_sub_div_out_tensors)

        negative_one = [-1.0]
        negative_one_vector=[]
        negative_one_vector += bytearray(struct.pack("f", negative_one[0]))
        self.buildBuffer("negative_one_buffer", negative_one_vector)
        self.buildTensor([1],"negative_one_tensor",self.getBufferByName("negative_one_buffer"))

        w_sub_div_mul_out_tensors = []
        w_sub_div_mul_in_tensors = []
        w_sub_div_mul_in_tensors.append("w_sub_reciprocal_tensor")
        w_sub_div_mul_in_tensors.append("negative_one_tensor")
        self.buildTensor(self.shape,"w_sub_div_mul_tensor")
        w_sub_div_mul_out_tensors.append("w_sub_div_mul_tensor")
        self.buildOperatorCode("SGS_w_sub_div_mul",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_w_sub_div_mul",w_sub_div_mul_in_tensors,w_sub_div_mul_out_tensors)

        w_sub_div_mul_sub_out_tensors = []
        w_sub_div_mul_sub_in_tensors = []
        w_sub_div_mul_sub_in_tensors.append("w_sub_div_mul_tensor")
        w_sub_div_mul_sub_in_tensors.append("positive_one_tensor")
        self.buildTensor(self.shape,"w_sub_div_mul_sub_tensor")
        w_sub_div_mul_sub_out_tensors.append("w_sub_div_mul_sub_tensor")
        self.buildOperatorCode("SGS_w_sub_div_mul_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
        self.buildOperator("SGS_w_sub_div_mul_sub",w_sub_div_mul_sub_in_tensors,w_sub_div_mul_sub_out_tensors)

        w_multi0_out_tensors = []
        w_multi0_in_tensors = []
        w_multi0_in_tensors.append("w_sub_div_mul_sub_tensor")
        w_multi0_in_tensors.append("pw_tensor")
        self.buildTensor(self.shape,"w_multi0_tensor")
        w_multi0_out_tensors.append("w_multi0_tensor")
        self.buildOperatorCode("SGS_w_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_w_multi0",w_multi0_in_tensors,w_multi0_out_tensors)

        w_multi1_out_tensors = []
        w_multi1_in_tensors = []
        w_multi1_in_tensors.append("w_multi0_tensor")
        w_multi1_in_tensors.append("sw_tensor")
        self.buildTensor(self.shape,"w_multi1_tensor")
        w_multi1_out_tensors.append("w_multi1_tensor")
        self.buildOperatorCode("SGS_w_multi1",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_w_multi1",w_multi1_in_tensors,w_multi1_out_tensors)


        h_sub_out_tensors = []
        h_sub_in_tensors = []
        h_sub_in_tensors.append("th_tensor")
        h_sub_in_tensors.append("positive_one_tensor")
        self.buildTensor(self.shape,"h_sub_tensor")
        h_sub_out_tensors.append("h_sub_tensor")
        self.buildOperatorCode("SGS_h_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
        self.buildOperator("SGS_h_sub",h_sub_in_tensors,h_sub_out_tensors)

        h_sub_div_out_tensors = []
        h_sub_div_in_tensors = []
        h_sub_div_in_tensors.append("h_sub_tensor")
        self.buildTensor(self.shape,"h_sub_reciprocal_tensor")
        h_sub_div_out_tensors.append("h_sub_reciprocal_tensor")
        cus_code = 'Reciprocal'
        cus_options = [(b"numeratort",1,"int")]
        options = self.createFlexBuffer(cus_options)
        self.buildOperatorCode("SGS_h_sub_reciprocal",self.BuiltinOperator.CUSTOM, cus_code)
        self.buildOperator("SGS_h_sub_reciprocal",h_sub_div_in_tensors,h_sub_div_out_tensors)

        h_sub_div_mul_out_tensors = []
        h_sub_div_mul_in_tensors = []
        h_sub_div_mul_in_tensors.append("h_sub_reciprocal_tensor")
        h_sub_div_mul_in_tensors.append("negative_one_tensor")
        self.buildTensor(self.shape,"h_sub_div_mul_tensor")
        h_sub_div_mul_out_tensors.append("h_sub_div_mul_tensor")
        self.buildOperatorCode("SGS_h_sub_div_mul",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_h_sub_div_mul",h_sub_div_mul_in_tensors,h_sub_div_mul_out_tensors)

        h_sub_div_mul_sub_out_tensors = []
        h_sub_div_mul_sub_in_tensors = []
        h_sub_div_mul_sub_in_tensors.append("h_sub_div_mul_tensor")
        h_sub_div_mul_sub_in_tensors.append("positive_one_tensor")
        self.buildTensor(self.shape,"h_sub_div_mul_sub_tensor")
        h_sub_div_mul_sub_out_tensors.append("h_sub_div_mul_sub_tensor")
        self.buildOperatorCode("SGS_h_sub_div_mul_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
        self.buildOperator("SGS_h_sub_div_mul_sub",h_sub_div_mul_sub_in_tensors,h_sub_div_mul_sub_out_tensors)

        h_multi0_out_tensors = []
        h_multi0_in_tensors = []
        h_multi0_in_tensors.append("h_sub_div_mul_sub_tensor")
        h_multi0_in_tensors.append("ph_tensor")
        self.buildTensor(self.shape,"h_multi0_tensor")
        h_multi0_out_tensors.append("h_multi0_tensor")
        self.buildOperatorCode("SGS_h_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_h_multi0",h_multi0_in_tensors,h_multi0_out_tensors)

        h_multi1_out_tensors = []
        h_multi1_in_tensors = []
        h_multi1_in_tensors.append("h_multi0_tensor")
        h_multi1_in_tensors.append("sh_tensor")
        self.buildTensor(self.shape,"h_multi1_tensor")
        h_multi1_out_tensors.append("h_multi1_tensor")
        self.buildOperatorCode("SGS_h_multi1",tflite.BuiltinOperator.BuiltinOperator().MUL)
        self.buildOperator("SGS_h_multi1",h_multi1_in_tensors,h_multi1_out_tensors)

        x1_out_tensors = []
        x1_in_tensors = []
        x1_in_tensors.append("x_multi1_tensor")
        x1_in_tensors.append("w_multi1_tensor")
        self.buildTensor(self.shape,"x1_tensor")
        x1_out_tensors.append("x1_tensor")
        self.buildOperatorCode("SGS_x1_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
        self.buildOperator("SGS_x1_sub",x1_in_tensors,x1_out_tensors)

        y1_out_tensors = []
        y1_in_tensors = []
        y1_in_tensors.append("y_multi1_tensor")
        y1_in_tensors.append("h_multi1_tensor")
        self.buildTensor(self.shape,"y1_tensor")
        y1_out_tensors.append("y1_tensor")
        self.buildOperatorCode("SGS_y1_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
        self.buildOperator("SGS_y1_sub",y1_in_tensors,y1_out_tensors)

        x2_out_tensors = []
        x2_in_tensors = []
        x2_in_tensors.append("x_multi1_tensor")
        x2_in_tensors.append("w_multi1_tensor")
        self.buildTensor(self.shape,"x2_tensor")
        x2_out_tensors.append("x2_tensor")
        self.buildOperatorCode("SGS_x2_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
        self.buildOperator("SGS_x2_add",x2_in_tensors,x2_out_tensors)

        y2_out_tensors = []
        y2_in_tensors = []
        y2_in_tensors.append("y_multi1_tensor")
        y2_in_tensors.append("h_multi1_tensor")
        self.buildTensor(self.shape,"y2_tensor")
        y2_out_tensors.append("y2_tensor")
        self.buildOperatorCode("SGS_y2_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
        self.buildOperator("SGS_y2_add",y2_in_tensors,y2_out_tensors)
        return x1_out_tensors + y1_out_tensors + x2_out_tensors + y2_out_tensors


if __name__ == '__main__':
    None
