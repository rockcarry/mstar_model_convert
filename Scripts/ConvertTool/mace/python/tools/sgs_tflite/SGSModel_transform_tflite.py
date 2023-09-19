import numpy as np
import six
import math
import pdb
import copy
from mace.proto import mace_pb2
from mace.python.tools.convert_util import mace_check
from mace.python.tools.converter_tool.base_converter import MaceKeyword

class TransformToSchema(object):
    """A class for transform mace model to schema model.
    """
    def __init__(self,  model,convert_kb=None):
        self._splitOP_transform = {
            'Conv2D':self.split_Conv2D,
            'DepthwiseConv2d':self.split_DepthwiseConv2d,
            'Reshape':self.split_Reshape,
            'Softmax':self.split_Softmax,
            'Pooling':self.split_Pooling,
            'Eltwise':self.split_Eltwise,
            'Concat':self.split_Concat,
            'Cos':self.split_Cos,
            'Pad':self.split_Pad,
            'FullyConnected':self.split_FullyConnected,
            'Reduce':self.split_Reduce_Mean,
            'Transpose':self.split_Transpose,
            'Split':self.split_Split,
            'Split_V':self.split_Split_V,
            'BatchToSpaceND':self.split_BatchToSpaceND,
            'SpaceToBatchND':self.split_SpaceToBatchND,
            'ArgMax':self.split_ArgMax,
            'ResizeBilinear':self.split_ResizeBilinear,
            'Activation':self.split_Activation,
            'Exp':self.split_Exp,
            'FloorDiv':self.split_FloorDiv,
            'PostProcess_Unpack':self.split_PostProcess_Unpack,
            'TFLite_Detection_NMS':self.split_TFLite_Detection_NMS,
            'Custom': self.split_Custom,
            'TransposeConv': self.split_TransposeConv,
            'Slice':self.split_Slice,
            'Abs':self.split_Abs,
            'Sqrt':self.split_Sqrt,
            'Square':self.split_Square,
            'Rsqrt':self.split_Rsqrt,
            'Tile':self.split_Tile,
            'Pack':self.split_Pack,
            'Unpack':self.split_Unpack,
            'StridedSlice':self.split_StridedSlice,
            'Sin':self.split_Sin,
            'BatchMatmul':self.split_BatchMatmul,
            'FakeOp':self.split_FakeOp,
            'ResizeNearestNeighbor':self.split_ResizeNearestNeighbor,
            'Cast':self.split_Cast,
            'Shape':self.split_Shape,
            'Less':self.split_Less,
            'Log':self.split_Log,
            'Greater':self.split_Greater,
            'GreaterEqual':self.split_Greater_equal,
            'Equal':self.split_Equal,
            'Round':self.split_Round,
            'Gather':self.split_Gather,
            'Topkv2':self.split_Topkv2
        }
        self._SGSModel = model
        self._convert_kb = convert_kb
        self._oriModel = copy.deepcopy(model)
        self._maceOpArray = np.array([])
        self.name_shape_map = {}
        self.finetuneArray = []

    def createDynamicTensors(self, SGSModel):
        for tensor in SGSModel.tensors:
          self.name_shape_map[tensor.name] = tensor.dims

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

    def run(self):
        self.createDynamicTensors(self._SGSModel)
        for op in self._oriModel.op:
          type = op.type
          self.infect_tensor_transpose_setting(op)
          self._splitOP_transform[type](op)

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
    def add_tensor(self, net, name, shape, data_type, value ,data_fromat = mace_pb2.DT_NHWC):
        #check dumplicated name
        for tensor in net.tensors:
               if tensor.name == name:
                 six.print_("find dumplicated tensor name",name)
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

    def get_axes_by_shape(self, axis,dims):
        axis = dims + axis if axis < 0 else axis
        axis_result = 0 if dims == 4 else axis
        if dims == 4:
          if axis == 1:
            axis_result = 3
          elif axis == 2:
            axis_result = 1
          elif axis == 3:
            axis_result = 2
        return axis_result

    def convert_const_tensor(self,const_tensor_name):
        def SGS_Conv_Laplace(beta, seed):
              fData1 = SGS_Conv_Uniform(0.0, 1.0, seed)
              fData2 = SGS_Conv_Uniform(0.0, 1.0, seed)
              if(fData1 <= 0.5):
                  fRet = -beta * math.log(1.0 - fData2)
              else:
                  fRet = beta * math.log(fData2)
              return fRet
        def SGS_Conv_Uniform(a, b, seed):
            seed = 2045 * seed + 1
            seed = seed - (seed / 1048576)
            t = seed / 1048576.0
            t = a + (b - a) * t
            return t
        const_tensor_1 = []
        const_tensor = self.find_tensor_by_name(const_tensor_name)
        if const_tensor.data_type == mace_pb2.DT_FLOAT:
            if len(const_tensor.float_data) != 0:
                const_tensor_ndarray = np.array(const_tensor.float_data)
                const_tensor_ndarray = const_tensor_ndarray.flatten()
                for i in range(len(const_tensor_ndarray)):
                    const_tensor_ndarray[i] = SGS_Conv_Laplace(25.0, const_tensor_ndarray[i])
                    if((const_tensor_ndarray[i] % 4 == 0) or (const_tensor_ndarray[i] % 6 == 0) or  (const_tensor_ndarray[i] % 7 == 0)):
                        const_tensor_ndarray[i] *= -1
                    const_tensor_ndarray[i] = float(const_tensor_ndarray[i]) / 100.0
                const_tensor_1 = list(const_tensor_ndarray)
        convert_const_name = '_convert#' + const_tensor_name
        self.add_tensor(self._SGSModel, convert_const_name, const_tensor.dims,
                                mace_pb2.DT_FLOAT,const_tensor_1)
        return convert_const_name

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
                   op_.type = "PRELU"
                   self._maceOpArray = np.append(self._maceOpArray,op_)
                 if a_type.decode() == 'LOGISTIC':
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
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                    name = arg[i].name
                    if name == MaceKeyword.mace_output_type_str:
                        OutputType = arg[i].i
                op_.type = "ARG_MAX"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_BatchToSpaceND(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "BATCH_TO_SPACE_ND"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_SpaceToBatchND(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = "SPACE_TO_BATCH_ND"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_BatchNorm(self, op):
        pass

    def split_Concat(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CONCATENATION"
            self._maceOpArray = np.append(self._maceOpArray,op_)
            outputName = op_.output[0]
            outputTensor = self.find_tensor_by_name(outputName)
            for idx, arg in enumerate(op_.arg):
                if arg.name == MaceKeyword.mace_axis_str:
                    if arg.i < 0:
                        arg.i = len(outputTensor.dims) + arg.i

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
                  elif name == "RELU6":
                    fusedActivationFunction = arg[i].i
                  elif name == "RELU":
                    fusedActivationFunction = arg[i].i
                  elif name == "RELU_N1_TO_1":
                    fusedActivationFunction = arg[i].i
                  elif name == "TANH":
                    fusedActivationFunction = arg[i].i
                  elif name == "SIGN_BIT":
                    fusedActivationFunction = arg[i].i
                inputTensor = self.find_tensor_by_name(xi)
                if group == 1:
                    op_.type = "CONV_2D"
                    self._maceOpArray = np.append(self._maceOpArray,op_)
                else:
                    #TODO
                    mace_check(group != 1,"Conv2D group > 1 unsupport yet")
                if self._convert_kb == True:
                    #automatically change const tensor data
                    op_.input[1] = self.convert_const_tensor(op_.input[1])


    def split_Deconv2D(self, op):
        #TRANSPOSE_CONV
        pass

    def split_DepthwiseConv2d(self, op):
      for op_ in self._SGSModel.op:
          if op_ == op:
            op_name = op_.name
            input_data_name = op_.input[0]
            filter_name = op_.input[1]
            biase_name = op_.input[2]
            filter_tensor = self.find_tensor_by_name(op_.input[1])
            [n,c,h,w] = filter_tensor.dims[:]
            if filter_tensor.float_data == [] and filter_tensor.int32_data == []:
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
            op_.type = "DEPTHWISE_CONV_2D"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_DepthToSpace(self, op):
        pass

    def split_Expand(self, op):
        pass

    def split_Eltwise(self, op):
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
                elif type == 3:
                  op_.type = 'DIV'
                elif type == 7:
                  op_.type = 'ABS'
                else:
                  six.print_("eltwise do not support")
              elif name == "RELU6":
                fusedActivationFunction = arg[i].i
              elif name == "RELU":
                fusedActivationFunction = arg[i].i
              elif name == "RELU_N1_TO_1":
                fusedActivationFunction = arg[i].i
              elif name == "TANH":
                fusedActivationFunction = arg[i].i
              elif name == "SIGN_BIT":
                fusedActivationFunction = arg[i].i
            self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Exp(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "EXP"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_FullyConnected(self, op):
        for op_ in self._SGSModel.op:
           if op_ == op:
             op_.type = "FULLY_CONNECTED"
             self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Gather(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            inputshape = self.get_shape_by_name(op_.input[0])
            arg = op_.arg
            for i in six.moves.range(len(arg)):
              name = arg[i].name
              if name == MaceKeyword.mace_axis_str:
                arg[i].i = self.get_axes_by_shape (arg[i].i, len(inputshape))
            op_.type = "GATHER"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_LSTM(self, op):
        pass

    def split_Pad(self, op):
       xi = op.input[1]
       inputTensor = self.find_tensor_by_name(xi)
       for op_ in self._SGSModel.op:
          if op_ == op:
             op_.type = "PAD"
             self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Pooling(self, op):
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        kernel_h,kernel_w,pad,stride = -1,-1,0,0
        for op_ in self._SGSModel.op:
            if op_ == op:
              arg = op_.arg
              for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_pooling_type_str:
                  pooling_type = arg[i].i
                  if pooling_type == 1:
                    op_.type = 'AVERAGE_POOL_2D'
                  elif pooling_type == 2:
                    op_.type = 'MAX_POOL_2D'
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Reduce(self, op):
        xi = op.input[0]
        op_name = op.name
        input_shape = self.get_shape_by_name(op.input[0])
        inputTensor = self.find_tensor_by_name(xi)
        output_shape = op.output_shape[0].dims[:]
        arg = op.arg
        axis = -1
        reduce_type = "Unkonw"
        for idx in six.moves.range(len(arg)):
          name = arg[idx].name
          if name == MaceKeyword.mace_reduce_type_str:
            type = arg[idx].i
            # MEAN = 0 MIN = 1 MAX = 2 PROD = 3 SUM = 4
            if type == 0:
               reduce_type = "MEAN"
            elif type == 2:
               reduce_type = 'REDUCE_MAX'
            elif type == 4:
               reduce_type = 'SUM'
            else:
               mace_check(False,'Reduce do not support.')
        for op_ in self._SGSModel.op:
            if op_ == op:
                op_.type = reduce_type
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Reshape(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "RESHAPE"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Slice(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "SLICE"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_StridedSlice(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "STRIDED_SLICE"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_BatchMatmul(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "BATCH_MATMUL"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Sqrt(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "SQRT"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Rsqrt(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "RSQRT"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Sin(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "SIN"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Cos(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "COS"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Log(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "LOG"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Square(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "SQUARE"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Cast(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CAST"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Abs(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "ABS"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Split(self, op):
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        if inputTensor.data_format == mace_pb2.DT_NHWC:
          for op_ in self._SGSModel.op:
            if op_ == op:
              arg = op_.arg
              for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_num_split_str:
                  NumSplit = arg[i].i
              op_.type = "SPLIT"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Split_V(self, op):
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        if inputTensor.data_format == mace_pb2.DT_NHWC:
          for op_ in self._SGSModel.op:
            if op_ == op:
              arg = op_.arg
              for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_num_split_str:
                  NumSplit = arg[i].i
              op_.type = "SPLIT_V"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Softmax(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            xi = op.input[0]
            op_name = op.name
            slice_point_enable = False
            inputTensor = self.find_tensor_by_name(xi)
            inputShape = self.get_shape_by_name(xi)
            output_shape = op.output_shape[0].dims
            op_.type = "SOFTMAX"
            mace_check(inputTensor.data_format == mace_pb2.DT_NHWC, "Invalid data format!! ")
            for outputName in op_.output:
              outputTensor = self.find_tensor_by_name(outputName)
              outputTensor.data_format = mace_pb2.DT_NHWC
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Transpose(self, op):
        op_name = op.name
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        input_shape = self.get_shape_by_name(xi)
        intputShapeSize = len(input_shape)
        output_shape = op.output_shape[0].dims
        if inputTensor.data_format == mace_pb2.DT_NHWC:
          for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "TRANSPOSE"
              self._maceOpArray = np.append(self._maceOpArray,op_)
        else:
          mace_check(inputTensor.data_format == mace_pb2.DT_NHWC, "Tflite only support NHWC invalid data format!!")

    def split_Tile(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "TILE"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Pack(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "PACK"
            self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Unpack(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "UNPACK"
            self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Custom(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "CUSTOM"
            self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Normalize(self, op):
        pass
    def split_Reorg(self, op):
        pass
    def split_Upsample(self, op):
        pass
    def split_Greater_equal(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "GREATER_EQUAL"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Greater(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "GREATER"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Reduce_Mean(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              #op_name = op_.name
              op_name = op_.name + op_.input[0]
              output_shape = op_.output_shape[0].dims[:]
              arg = op_.arg

              for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_reduce_type_str:
                    type = arg[i].i

              # mean
              if type == 0:
                input_shape = self.get_shape_by_name(op_.input[0])
                axis_tensor = self.find_tensor_by_name(op.input[1])
                axis_data = axis_tensor.int32_data

                if len(input_shape) == 4 and len(output_shape) == 4:
                  # SGS MEAN only support H W
                  if axis_data == [1,2]:
                      op_.type = "MEAN"
                      self._maceOpArray = np.append(self._maceOpArray,op_)
                      return
                  #reducemean = reduceSum % dimNum
                  else:
                      # creat axis tensor
                      axis_tensor_name = op_name + op_.input[0] + '_axis'
                      axis_tensor_data = axis_data
                      axis_tensor_shape = [len(axis_data)]
                      self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                                      mace_pb2.DT_INT32, axis_tensor_data)
                      op_reduceSum = self._SGSModel.op.add()
                      op_reduceSum.name = op_name + '_reduceSum'
                      op_reduceSum.type = 'SUM'
                      op_reduceSum.input.extend([op_.input[0]])
                      op_reduceSum.input.extend([axis_tensor_name])
                      output_op_reduceSum = op_reduceSum.name + op_.output[0] + '_output'
                      op_reduceSum.output.extend([output_op_reduceSum])
                      tmp_shape = output_shape
                      op_reduceSum.output_shape.add()
                      op_reduceSum.output_shape[0].dims.extend(tmp_shape)
                      self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)
                      #creat div
                      for i in range(len(axis_data)):
                          axis_new = axis_data[i]
                          if i == 0:
                              reduce_dim = input_shape[axis_new]
                          else:
                              reduce_dim *= input_shape[axis_new]
                      const_tensor_name = op_name + op_.input[0] + '_const'
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
                      op_div.output_shape.extend(op_.output_shape)
                      self._maceOpArray = np.append(self._maceOpArray,op_div)
                      self.remove_op_with_name(op_name)
                      return
                else:
                      # creat axis tensor
                      #axis_tensor_name = op_name + op_.input[0] + '_axis'
                      axis_tensor_name = op_name + '_axis'
                      axis_tensor_data = axis_data
                      axis_tensor_shape = [len(axis_data)]
                      self.add_tensor(self._SGSModel, axis_tensor_name, axis_tensor_shape,
                      mace_pb2.DT_INT32, axis_tensor_data)
                      op_reduceSum = self._SGSModel.op.add()
                      op_reduceSum.name = op_name + '_reduceSum'
                      op_reduceSum.type = 'SUM'
                      op_reduceSum.input.extend([op_.input[0]])
                      op_reduceSum.input.extend([axis_tensor_name])
                      output_op_reduceSum = op_reduceSum.name + op_.output[0] + '_output'
                      op_reduceSum.output.extend([output_op_reduceSum])
                      tmp_shape = output_shape
                      op_reduceSum.output_shape.add()
                      op_reduceSum.output_shape[0].dims.extend(tmp_shape)
                      self._maceOpArray = np.append(self._maceOpArray,op_reduceSum)
                      #creat div
                      for i in range(len(axis_data)):
                          axis_new = axis_data[i]
                          if i == 0:
                              reduce_dim = input_shape[axis_new]
                          else:
                              reduce_dim *= input_shape[axis_new]
                      const_tensor_name = op_name + op_.input[0] + '_const'
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
                      op_div.output_shape.extend(op_.output_shape)
                      self._maceOpArray = np.append(self._maceOpArray,op_div)
                      self.remove_op_with_name(op_name)
                      return
              # sum
              elif type == 4:
                op_.type = "SUM"
                self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_ResizeBilinear(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              arg = op_.arg
              for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_align_corners_str:
                  AlignCorners = arg[i].i
              op_.type = "RESIZE_BILINEAR"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_ResizeNearestNeighbor(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              arg = op_.arg
              for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_align_corners_str:
                  AlignCorners = arg[i].i
                elif name == MaceKeyword.mace_HalfPixelCenters_str:
                  HalfPixelCenters = arg[i].i
              op_.type = "RESIZE_NEAREST_NEIGHBOR"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_TransposeConv(self, op):
         for op_ in self._SGSModel.op:
           if op_ == op:
             op_.type = "TRANSPOSE_CONV"
             self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_PostProcess_Unpack(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.name = 'PostProcess_Unpack' + op.name
            op_.type = "CUSTOM"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_TFLite_Detection_NMS(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.name = 'TFLite_Detection_NMS' + op.name
            op_.type = "CUSTOM"
            self._maceOpArray = np.append(self._maceOpArray,op_)


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
    def split_FakeOp(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.name = 'Image_Concatenation'+ op.name
            op_.type = "CUSTOM"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Shape(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "SHAPE"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Less(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "LESS"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Equal(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "EQUAL"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Round(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "ROUND"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Gather(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "GATHER"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Topkv2(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "TOPK_V2"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_FloorDiv(self, op):
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "FLOOR_DIV"
            self._maceOpArray = np.append(self._maceOpArray,op_)