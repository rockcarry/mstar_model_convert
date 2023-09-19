# -*- coding: utf-8 -*-

import os
import sys
import ctypes
import struct
import numpy as np
import calibrator_custom
from collections import OrderedDict
import pdb
from . import chalk_common
from functools import reduce

if 'IPU_TOOL' in os.environ:
    Project_path = os.environ['IPU_TOOL']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/ConvertTool"))
elif 'SGS_IPU_DIR' in os.environ:
    Project_path = os.environ['SGS_IPU_DIR']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/ConvertTool"))
else:
    raise OSError('Run `source cfg_env.sh` in top directory.')

from third_party import tflite
from third_party.python import flatbuffers

BUILDER = flatbuffers.Builder(0)
BUFFERS = []
TENSORS = OrderedDict()
TENSOR_MINMAX = []
OPERATORCODES = []
OPERATORS = []
SUBGRAPHS = []


class SGS_Builder(object):
    def __init__(self):
        self.LIB = ctypes.cdll.LoadLibrary(os.path.join(Project_path, "libs/x86_64/libSGSCusOP.so"))

    @staticmethod
    def convert2struct(dtype):
        if dtype == 'uint8':
            return ('B', tflite.TensorType.TensorType().UINT8)
        elif dtype == 'int16':
            return ('h', tflite.TensorType.TensorType().INT16)
        elif dtype == 'float32':
            return ('f', tflite.TensorType.TensorType().FLOAT32)
        elif dtype == 'int32':
            return ('i', tflite.TensorType.TensorType().INT32)
        elif dtype == 'int64':
            return ('q', tflite.TensorType.TensorType().INT64)
        elif dtype == 'complex64':
            return ('ff', tflite.TensorType.TensorType().COMPLEX64)
        else:
            raise ValueError('Not support data type:', dtype)

    @staticmethod
    def checkTensorNameValid(name, prefix='Tensor'):
        if name in TENSORS.keys() or name is None:
            num = 0
            while True:
                if prefix + '_{}'.format(num) in TENSORS.keys():
                    num += 1
                else:
                    return prefix + '_{}'.format(num)
        else:
            return name

    @staticmethod
    def findTensorMinmaxByName(name):
        for idx, tensor_minmax in enumerate(TENSOR_MINMAX):
            if tensor_minmax['name'] == name:
                return tensor_minmax, idx
        return None, None

    @staticmethod
    def resetContainer():
        BUFFERS.clear()
        TENSORS.clear()
        TENSOR_MINMAX.clear()
        OPERATORCODES.clear()
        OPERATORS.clear()
        SUBGRAPHS.clear()

class Tensor(SGS_Builder):
    def __init__(self, data=None, shape=None, dtype='float32', name=None, prefix='Tensor',
                bit=16, minimum=None, maximum=None):
        super(Tensor, self).__init__()
        self.np_data = data
        self.shape = tuple(shape) if shape is not None else None
        self.dtype = dtype
        self.name = self.checkTensorNameValid(name, prefix)
        self.tensor_id = None
        self.is_const = None
        self.minmax = OrderedDict()
        if data is None and shape is None:
            raise ValueError('data or shape must be set!')
        self._createTensor()
        self.minmax['name'] = self.name
        self.minmax['bit'] = bit
        minimum_default = -32767 if minimum is None else minimum
        maximum_default = 32767 if maximum is None else maximum
        self.minmax['min'] = minimum_default if isinstance(minimum_default, list) else [minimum_default]
        self.minmax['max'] = maximum_default if isinstance(maximum_default, list) else [maximum_default]

    def _createBuffer(self):
        if len(BUFFERS) == 0:
            # the 0th entry of this array must be an empty buffer (sentinel).
            tflite.Buffer.BufferStartDataVector(BUILDER, 0)
            firstData = BUILDER.EndVector(0)
            tflite.Buffer.BufferStart(BUILDER)
            tflite.Buffer.BufferAddData(BUILDER, firstData)
            firstEmptyBuffer = tflite.Buffer.BufferEnd(BUILDER)
            BUFFERS.append(firstEmptyBuffer)

        if self.np_data is None:
            tflite.Buffer.BufferStartDataVector(BUILDER, 0)
            data = BUILDER.EndVector(0)
            self.is_const = False
        else:
            if not self.shape is None and self.shape != self.np_data.shape:
                raise ValueError('shape and data.shape not same! ({} vs {})'.format(self.shape, self.np_data.shape))
            self.shape = self.np_data.shape
            self.dtype = str(self.np_data.dtype)
            if self.dtype == 'complex64':
                cmplx_data = np.ones(self.np_data.size * 2)
                cmplx_data[::2] = self.np_data.real.flatten()
                cmplx_data[1::2] = self.np_data.imag.flatten()
                databytes = struct.pack(self.convert2struct(self.dtype)[0] * len(self.np_data.flatten().tolist()),
                    *cmplx_data.tolist())
            else:
                databytes = struct.pack(self.convert2struct(self.dtype)[0] * len(self.np_data.flatten().tolist()),
                    *self.np_data.flatten().tolist())
            tflite.Buffer.BufferStartDataVector(BUILDER, len(databytes))
            for i in reversed(databytes):
                BUILDER.PrependByte(i)
            data = BUILDER.EndVector(len(databytes))
            self.is_const = True

        tflite.Buffer.BufferStart(BUILDER)
        tflite.Buffer.BufferAddData(BUILDER, data)
        BUFFERS.append(tflite.Buffer.BufferEnd(BUILDER))
        return len(BUFFERS) - 1

    def _createTensor(self):
        buffer_id = self._createBuffer()
        tensor_name = BUILDER.CreateString(self.name)
        tflite.Tensor.TensorStartShapeVector(BUILDER, len(self.shape))
        for i in reversed(self.shape):
            BUILDER.PrependInt32(i)
        shapes = BUILDER.EndVector(len(self.shape))
        tflite.Tensor.TensorStart(BUILDER)
        tflite.Tensor.TensorAddShape(BUILDER, shapes)
        tflite.Tensor.TensorAddType(BUILDER, self.convert2struct(self.dtype)[1])
        tflite.Tensor.TensorAddBuffer(BUILDER, buffer_id)
        tflite.Tensor.TensorAddName(BUILDER, tensor_name)
        tflite.Tensor.TensorAddIsVariable(BUILDER, False)
        TENSORS[self.name] = tflite.Tensor.TensorEnd(BUILDER)
        self.tensor_id = len(TENSORS) - 1

    def __repr__(self):
        ss = '{}'.format(', '.join(str(v) for v in self.shape))
        return '<Tensor \'{}\' shape=({}) dtype={}>'.format(self.name, ss, self.dtype)


class Operator(SGS_Builder):
    def __init__(self, inputs, outputs, builtin_code, custom_code=b'Builtin', builtin_options_type=None,
                builtin_options=None, custom_options=None):
        super(Operator, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.builtin_code = builtin_code
        self.custom_code = custom_code
        self.builtin_options_type = builtin_options_type
        self.builtin_options = builtin_options
        self.custom_options = custom_options
        self.operator_id = None
        self._createOperator()
        self._checkMinMax()

    def _createOpcodeByIPUversion(self):
        custom_code_str = BUILDER.CreateString(self.custom_code)
        tflite.OperatorCode.OperatorCodeStart(BUILDER)
        if calibrator_custom.utils.get_sdk_version() in ['1', 'Q_0']:
            if self.builtin_code > 127:#schema 1.14 The maximum number of operators supported is 128(0~127)
                chalk_common.mace_check(0, 'Model schema version is higher than 1.14, Please check model !! \n ')
            tflite.OperatorCode.OperatorCodeAddDeprecatedBuiltinCode(BUILDER, self.builtin_code)
            tflite.OperatorCode.OperatorCodeAddCustomCode(BUILDER, custom_code_str)
            tflite.OperatorCode.OperatorCodeAddVersion(BUILDER, 1)
        else:
            tflite.OperatorCode.OperatorCodeAddBuiltinCode(BUILDER, self.builtin_code)
            tflite.OperatorCode.OperatorCodeAddCustomCode(BUILDER, custom_code_str)
            tflite.OperatorCode.OperatorCodeAddVersion(BUILDER, 1)
        OPERATORCODES.append(tflite.OperatorCode.OperatorCodeEnd(BUILDER))
        return len(OPERATORCODES) - 1

    def _createFlexBuffer(self, values):
        self.LIB.startCreatFlexBuffer.restype = None
        self.LIB.startCreatFlexBuffer()
        for value in values:
            if value[-1] == 'int':
                self.LIB.insertIntString.argtypes = [ctypes.c_char_p, ctypes.c_int]
                self.LIB.insertIntString.restype = None
                value_name = bytes(value[0])
                self.LIB.insertIntString(value_name, value[1])
            elif value[-1] == 'float':
                self.LIB.insertFloatString.argtypes = [ctypes.c_char_p, ctypes.c_float]
                self.LIB.insertFloatString.restype = None
                value_name = bytes(value[0])
                self.LIB.insertFloatString(value_name, value[1])
            else:
                raise ValueError('Only support Int and Float.')

        c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
        self.LIB.getFlexBufferData.restype = c_ubyte_p
        cusData = self.LIB.getFlexBufferData()

        self.LIB.getFlexBufferLenth.restype = ctypes.c_int
        bufferLen = self.LIB.getFlexBufferLenth()

        _allByteArray = bytearray()
        for i in range(bufferLen):
            _allByteArray.append(cusData[i])

        self.LIB.endCreatFlexBuffer.restype = None
        self.LIB.endCreatFlexBuffer()
        return _allByteArray

    def _createOperatorCode(self):
        custom_code_str = BUILDER.CreateString(self.custom_code)
        tflite.OperatorCode.OperatorCodeStart(BUILDER)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(BUILDER, self.builtin_code)
        tflite.OperatorCode.OperatorCodeAddCustomCode(BUILDER, custom_code_str)
        tflite.OperatorCode.OperatorCodeAddVersion(BUILDER, 1)
        OPERATORCODES.append(tflite.OperatorCode.OperatorCodeEnd(BUILDER))
        return len(OPERATORCODES) - 1

    def _createOperator(self):
        #opcode = self._createOperatorCode()
        opcode = self._createOpcodeByIPUversion()
        tflite.Operator.OperatorStartInputsVector(BUILDER, len(self.inputs))
        for in_tensor in reversed(self.inputs):
            BUILDER.PrependInt32(in_tensor.tensor_id)
        input_tensors = BUILDER.EndVector(len(self.inputs))

        tflite.Operator.OperatorStartOutputsVector(BUILDER, len(self.outputs))
        for out_tensor in reversed(self.outputs):
            BUILDER.PrependInt32(out_tensor.tensor_id)
        output_tensors = BUILDER.EndVector(len(self.outputs))

        cus_data = None
        if self.custom_options is not None:
            custom_options_flex = self._createFlexBuffer(self.custom_options)
            tflite.Operator.OperatorStartCustomOptionsVector(BUILDER, len(custom_options_flex))
            for meta_ubyte in reversed(custom_options_flex):
                BUILDER.PrependByte(meta_ubyte)
            cus_data = BUILDER.EndVector(len(custom_options_flex))

        tflite.Operator.OperatorStart(BUILDER)
        tflite.Operator.OperatorAddOpcodeIndex(BUILDER, opcode)
        tflite.Operator.OperatorAddInputs(BUILDER, input_tensors)
        tflite.Operator.OperatorAddOutputs(BUILDER, output_tensors)
        if self.builtin_options is not None:
            tflite.Operator.OperatorAddBuiltinOptionsType(BUILDER, self.builtin_options_type)
            tflite.Operator.OperatorAddBuiltinOptions(BUILDER, self.builtin_options)
        if cus_data is not None:
            tflite.Operator.OperatorAddCustomOptions(BUILDER, cus_data)
        OPERATORS.append(tflite.Operator.OperatorEnd(BUILDER))
        self.operator_id = len(OPERATORS) - 1

    def _checkMinMax(self):
        for idx, _ in enumerate(self.inputs):
            f, _ = self.findTensorMinmaxByName(self.inputs[idx].name)
            if f is None:
                if self.builtin_code in [tflite.BuiltinOperator.BuiltinOperator().CONV_2D,
                    tflite.BuiltinOperator.BuiltinOperator().CONV_3D,tflite.BuiltinOperator.BuiltinOperator().FULLY_CONNECTED] and idx == 1:
                    if len(self.inputs[idx].minmax['min']) != self.inputs[idx].shape[0]:
                        self.inputs[idx].minmax['min'] = self.inputs[idx].minmax['min'] * self.inputs[1].shape[0]
                    if len(self.inputs[idx].minmax['max']) != self.inputs[idx].shape[0]:
                        self.inputs[idx].minmax['max'] = self.inputs[idx].minmax['max'] * self.inputs[1].shape[0]
                elif self.builtin_code in [tflite.BuiltinOperator.BuiltinOperator().BATCH_MATMUL] and idx == 1:
                    NumMinMax = reduce(lambda x, y: x * y, self.inputs[idx].shape[:-1])
                    if len(self.inputs[idx].minmax['min']) != NumMinMax:
                        self.inputs[idx].minmax['min'] = self.inputs[idx].minmax['min'] * NumMinMax
                    if len(self.inputs[idx].minmax['max']) != NumMinMax:
                        self.inputs[idx].minmax['max'] = self.inputs[idx].minmax['max'] * NumMinMax
                else:
                    self.inputs[idx].minmax['min'] = self.inputs[idx].minmax['min'] * self.inputs[idx].shape[-1]
                    self.inputs[idx].minmax['max'] = self.inputs[idx].minmax['max'] * self.inputs[idx].shape[-1]
                TENSOR_MINMAX.append(self.inputs[idx].minmax)

        for idx, _ in enumerate(self.outputs):
            f, _ = self.findTensorMinmaxByName(self.outputs[idx].name)
            if f is None:
                if self.builtin_code in [tflite.BuiltinOperator.BuiltinOperator().BATCH_MATMUL]:
                    NumMinMax = reduce(lambda x, y: x * y, self.outputs[idx].shape[:-2]) * self.outputs[idx].shape[-1]
                    if len(self.outputs[idx].minmax['min']) != NumMinMax:
                        self.outputs[idx].minmax['min'] = self.outputs[idx].minmax['min'] * NumMinMax
                    if len(self.outputs[idx].minmax['max']) != NumMinMax:
                        self.outputs[idx].minmax['max'] = self.outputs[idx].minmax['max'] * NumMinMax
                else:
                    self.outputs[idx].minmax['min'] = self.outputs[idx].minmax['min'] * self.outputs[idx].shape[-1]
                    self.outputs[idx].minmax['max'] = self.outputs[idx].minmax['max'] * self.outputs[idx].shape[-1]
                TENSOR_MINMAX.append(self.outputs[idx].minmax)


class Model(SGS_Builder):
    def __init__(self, input_tensors, output_tensors, subgraph_name = 'SGS_Model'):
        super(Model, self).__init__()
        self.inputs = input_tensors if isinstance(input_tensors, list) else [input_tensors]
        self.outputs = output_tensors if isinstance(output_tensors, list) else [output_tensors]
        self.subgraph_name = subgraph_name
        self.model = None
        self._createSubGraph()
        self._createModel()

    def _createSubGraph(self):
        tflite.SubGraph.SubGraphStartTensorsVector(BUILDER, len(TENSORS))
        for tensor in reversed(TENSORS.values()):
            BUILDER.PrependUOffsetTRelative(int(tensor))
        tensors = BUILDER.EndVector(len(TENSORS))

        tflite.SubGraph.SubGraphStartOperatorsVector(BUILDER, len(OPERATORS))
        for operator in reversed(OPERATORS) :
            BUILDER.PrependUOffsetTRelative(int(operator))
        operators = BUILDER.EndVector(len(OPERATORS))

        tflite.SubGraph.SubGraphStartInputsVector(BUILDER, len(self.inputs))
        for in_tensor in reversed(self.inputs):
            BUILDER.PrependInt32(in_tensor.tensor_id)
        inputs = BUILDER.EndVector(len(self.inputs))

        tflite.SubGraph.SubGraphStartOutputsVector(BUILDER, len(self.outputs))
        for out_tensor in reversed(self.outputs):
            BUILDER.PrependInt32(out_tensor.tensor_id)
        outputs = BUILDER.EndVector(len(self.outputs))

        name = BUILDER.CreateString(self.subgraph_name)

        tflite.SubGraph.SubGraphStart(BUILDER)
        tflite.SubGraph.SubGraphAddTensors(BUILDER, tensors)
        tflite.SubGraph.SubGraphAddInputs(BUILDER, inputs)
        tflite.SubGraph.SubGraphAddOutputs(BUILDER, outputs)
        tflite.SubGraph.SubGraphAddOperators(BUILDER, operators)
        tflite.SubGraph.SubGraphAddName(BUILDER, name)
        SUBGRAPHS.append(tflite.SubGraph.SubGraphEnd(BUILDER))

    def _createModel(self):
        tflite.Model.ModelStartOperatorCodesVector(BUILDER, len(OPERATORCODES))
        for op_code in reversed(OPERATORCODES):
            BUILDER.PrependUOffsetTRelative(int(op_code))
        operator_codes = BUILDER.EndVector(len(OPERATORCODES))

        tflite.Model.ModelStartBuffersVector(BUILDER, len(BUFFERS))
        for buffer in reversed(BUFFERS):
            BUILDER.PrependUOffsetTRelative(int(buffer))
        buffers = BUILDER.EndVector(len(BUFFERS))

        tflite.Model.ModelStartOperatorCodesVector(BUILDER, len(SUBGRAPHS))
        for subgraph in SUBGRAPHS:
            BUILDER.PrependUOffsetTRelative(int(subgraph))
        subgraphs = BUILDER.EndVector(len(SUBGRAPHS))
        description = BUILDER.CreateString('Original Float')
        tflite.Model.ModelStart(BUILDER)
        tflite.Model.ModelAddVersion(BUILDER, 3)
        tflite.Model.ModelAddDescription(BUILDER, description)
        tflite.Model.ModelAddOperatorCodes(BUILDER, operator_codes)
        tflite.Model.ModelAddSubgraphs(BUILDER, subgraphs)
        tflite.Model.ModelAddBuffers(BUILDER, buffers)
        self.model = tflite.Model.ModelEnd(BUILDER)

    def save_buf(self):
        if calibrator_custom.versions.VERSION[:2] in ['1.', 'Q_']:
            file_identifier = b'TFL3'
        else:
            file_identifier = b'SIM2'
        BUILDER.Finish(self.model, file_identifier)
        buf = BUILDER.Output()
        self.resetContainer()
        return buf

    def save(self, model_path, input_config=None, convert_fixed=False, inputs=None):
        if calibrator_custom.utils.get_sdk_version() in ['1', 'Q_0']:
            file_identifier = b'TFL3'
        else:
            file_identifier = b'SIM2'
        BUILDER.Finish(self.model, file_identifier)
        buf = BUILDER.Output()

        debug_output_file = os.path.join(os.path.dirname(model_path), 'Debug_' + os.path.basename(model_path))
        float_output_file = model_path
        cmodel_float_file = calibrator_custom.utils.get_out_model_name(model_path, phase='CMD_Float')
        fixed_output_file = calibrator_custom.utils.get_out_model_name(model_path)
        compiler_config = calibrator_custom.utils.CompilerConfig()

        if input_config is None:
            with open(model_path, 'wb') as f:
                f.write(buf)
            print('Debug model saved in {}'.format(model_path))
            print('Indicate path of input_config.ini can auto convert to SGS Float Model.')
        else:
            if convert_fixed == False:
                with open(debug_output_file, 'wb') as f:
                    f.write(buf)
                print('Debug model saved in {}'.format(debug_output_file))
                float_converter = calibrator_custom.converter(debug_output_file, input_config)
                float_converter.convert(compiler_config.Debug2FloatConfig(), saved_path=float_output_file)
                print('Float model saved in {}'.format(float_output_file))

        if input_config is not None and convert_fixed and inputs is None:
            with open(debug_output_file, 'wb') as f:
                f.write(buf)
            print('Debug model saved in {}'.format(debug_output_file))
            float_converter = calibrator_custom.converter(debug_output_file, input_config)
            float_converter.update(TENSOR_MINMAX)
            float_converter.convert(compiler_config.Debug2FloatConfig(), saved_path=float_output_file)
            print('Float model saved in {}'.format(float_output_file))
            fixed_converter = calibrator_custom.calibrator(float_output_file, input_config)
            chalk_common.print_model(fixed_converter)
            fixed_converter.convert_cmodel_float(compiler_config.Float2CmodelFloatConfig(), saved_path=cmodel_float_file)
            fixed_converter.convert_fixed(compiler_config.CmodelFloat2FixedConfig(), fixed_output_file)
            print('Fixed model saved in {}'.format(fixed_output_file))

        if input_config is not None and convert_fixed and inputs is not None:
            with open(debug_output_file, 'wb') as f:
                f.write(buf)
            print('Debug model saved in {}'.format(debug_output_file))
            float_converter = calibrator_custom.converter(debug_output_file, input_config)
            float_converter.convert(compiler_config.Debug2FloatConfig(), saved_path=float_output_file)
            print('Float model saved in {}'.format(float_output_file))
            chalk_calibrator = chalk_common.Chalk_Calibrator(float_output_file, input_config)
            model_preprocess = chalk_common.autogen_preprocess(chalk_calibrator.model, input_config)
            training_input_formats = [pre['training_input_formats'] for pre in model_preprocess]
            preprocess_scripts = [pre['preprocess_script'] for pre in model_preprocess]
            print(chalk_calibrator)
            chalk_calibrator_inputs = chalk_calibrator.model.get_input_details()

            if inputs == 'RAWDATA':
                if 'BGR' in training_input_formats or \
                    'RGB' in training_input_formats or \
                    'GRAY' in training_input_formats:
                    raise ValueError('Please use image path for `inputs`')
                raw_inputs = []
                for idx, input_i in enumerate(chalk_calibrator_inputs):
                    data_file = 'input{}_data.npy'.format(idx)
                    if input_i['dtype'] == np.complex64:
                        data = np.random.rand(*input_i['shape']).astype(np.complex64)
                    else:
                        chalk_tensor = chalk_common.get_chalk_input_tensor(input_i['name'], self)
                        min_max = (chalk_tensor.minmax['min'][0], chalk_tensor.minmax['max'][0])
                        data = np.random.rand(*input_i['shape'])
                        data = (min_max[0] - min_max[1]) * data + min_max[1]
                        data = data.astype(np.float32)
                    np.save(data_file, data)
                    raw_inputs.append(data_file)
                input_list_file = 'input_list.txt'
                with open(input_list_file, 'w') as f:
                    f.write(','.join(raw_inputs))
                img_gen = chalk_common.chalk_image_generate(input_list_file, preprocess_scripts)
            else:
                img_gen = chalk_common.chalk_image_generate(inputs, preprocess_scripts)

            chalk_calibrator.convert(img_gen, fix_model=[fixed_output_file])
            print('\nFixed model saved in {}'.format(fixed_output_file))

        self.resetContainer()