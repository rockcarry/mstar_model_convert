import numpy as np
import six
import math
import pdb
import copy
from mace.proto import mace_pb2
from mace.python.tools.convert_util import mace_check
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.caffe import SGSModel_genAnchors
from mace.python.tools.caffe import SGSModel_prior_box

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
    def creatDynamicTensors(self, SGSModel):
        #1.collect all tensor (include constant and virable tensor)
        outputname_list = []
        outputShape_list = []
        #add constant tensor
        for tensor in SGSModel.tensors:
          tensor_name = tensor.name
          tensor_shape = tensor.dims[:]
          self.name_shape_map[tensor_name] = tensor_shape
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
          shape = self._inputShapeMap[name]
          if self._input_pack_model_arrays != 'None':
            if self._input_pack_model_arrays[0] == 'caffe':
                self._input_pack_model_arrays = copy.deepcopy(self._inputName)
            if name in self._input_pack_model_arrays:
               self.add_tensor(SGSModel, name, shape, mace_pb2.DT_FLOAT, None, mace_pb2.DT_NHWC)
          else:
            self.add_tensor(SGSModel, name, shape, mace_pb2.DT_FLOAT, None)
          self.name_shape_map[name] = shape
    def run(self):
        self.creatDynamicTensors(self._SGSModel)
        for op in self._oriModel.op:
          type = op.type
          self._splitOP_transform[type](op)
        self.finetuneNet()
        if self._input_pack_model_arrays != None or self._output_pack_model_arrays != None:
          self.changePackModel()
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
                six.print_("Remove identity: %s(%s)" % (op.name, op.type))
                self.safe_remove_node(op,
                                      self._producer.get(op.input[0], None))
                return True

        return False

    def remove_op_with_name(self,name):
        net = self._SGSModel
        for op in net.op:
            if op.name == name:
                six.print_("Remove op: %s(%s)" % (op.name, op.type))
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

    def changePackModel(self):
      #change output tensor to NCHW
      if self._output_pack_model_arrays != 'None':
          if self._output_pack_model_arrays[0] == "caffe":
            self._output_pack_model_arrays = copy.deepcopy(self._outputName)
          fillterTensor = []
          for op in self._SGSModel.op:
           for topName in op.output:
            for outputName in self._output_pack_model_arrays:
              if outputName == topName and (outputName not in fillterTensor):# find output op
                fillterTensor.append(topName)
                op_name = op.name
                outputTensor = self.find_tensor_by_name(outputName)
                output_shape = outputTensor.dims[:]
                if len(output_shape) == 4: #change PackModel to caffe
                    shadowTensor_name = outputTensor.name + "_shadow"
                    shadowTensor_data = []
                    shadowTensor_shape = output_shape
                    self.add_tensor(self._SGSModel ,shadowTensor_name, shadowTensor_shape,
                        outputTensor.data_type, shadowTensor_data)

                    op.output.remove(outputName)
                    op.output.extend([shadowTensor_name])
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
                    op_transpose.name = op_name + '_transpose'
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
                    shadowTensor_name = outputTensor.name + "_shadow"
                    shadowTensor_data = []
                    shadowTensor_shape = output_shape
                    self.add_tensor(self._SGSModel ,shadowTensor_name, shadowTensor_shape,
                        outputTensor.data_type, shadowTensor_data)

                    op.output.remove(outputName)
                    op.output.extend([shadowTensor_name])
                    #creat reshape
                    tmp_shape = [1,1,1,1]
                    output_len = len(output_shape)
                    for i in six.moves.range(output_len):
                      tmp_shape[3-i] = output_shape[output_len-1-i]
                    op_reshape = self._SGSModel.op.add()
                    op_reshape.name = op_name + '_reshape'
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
      #change input tensor to NCHW
      if self._input_pack_model_arrays != 'None':
        fillterTensor = []
        for op in self._SGSModel.op:
         for topName in op.input:
          for inputName in self._input_pack_model_arrays:
            if inputName == topName and (inputName not in fillterTensor):
             if op.type != 'TRANSPOSE':
               fillterTensor.append(topName)
               op_name = op.name
               outputTensor = self.find_tensor_by_name(inputName)
               input_shape = self.name_shape_map[inputName]
               if len(input_shape) == 4:
                   # creat transpose for NCHW to NHWC
                   op_transpose = self._SGSModel.op.add()
                   op_transpose.name = op_name + '_shadow'
                   op_transpose.type = 'TRANSPOSE'
                   shape_tensor_name = op_transpose.name + '_shape'
                   shape_tensor_data = [0,2,3,1]
                   shape_tensor_shape = [4]
                   self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
                       mace_pb2.DT_INT32, shape_tensor_data)
                   op_transpose.input.extend([inputName])
                   op_transpose.input.extend([shape_tensor_name])
                   output_op_transpose = op_transpose.name + '_output'
                   op_transpose.output.extend([output_op_transpose])
                   tmp_dim = [0,1,2,3]
                   tmp_shape = [0,1,2,3]
                   for i in six.moves.range(len(tmp_dim)):
                     tmp_shape[i] = input_shape[tmp_dim[i]]
                   op_transpose.output_shape.add()
                   op_transpose.output_shape[0].dims.extend(tmp_shape)
                   self._maceOpArray = np.append(self._maceOpArray,op_transpose)

                   op.input.remove(inputName)
                   op.input.extend([output_op_transpose])

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
                 if a_type.decode() == 'SIGMOID':
                   op_.type = "LOGISTIC"
                   self._maceOpArray = np.append(self._maceOpArray,op_)
                 if a_type.decode() == 'TANH':
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
                arg[i].i = len(inputShape) + arg[i].i if arg[i].i < 0 else arg[i].i
                if len(output_shape[0].dims) == 4:#nchw -> nhwc
                  if arg[i].i == 1:
                    arg[i].i = 3
                  elif arg[i].i == 2:
                    arg[i].i = 1
                  elif arg[i].i == 3:
                    arg[i].i = 2
                axis = arg[i].i
            #add axis tensor into input arrays
            axis_tensor_name = op.name + '_axis'
            axis_data = [axis]
            axis_data_shape = [1]
            self.add_tensor(self._SGSModel, axis_tensor_name, axis_data_shape,
              mace_pb2.DT_INT32, axis_data)
            op_.input.extend([axis_tensor_name])
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Axpy(self, op):
        '''
        * @param Formulation:
        *            F = a * X + Y
        *    Shape info:
        *            a:  N x C          --> bottom[0]
        *            X:  N x C x H x W  --> bottom[1]
        *            Y:  N x C x H x W  --> bottom[2]
        *            F:  N x C x H x W  --> top[0]
        '''
        name = op.name
        a = op.input[0]
        X = op.input[1]
        Y = op.input[2]
        a_shape = self.get_shape_by_name(a)
        #creat Mul
        op_mul = self._SGSModel.op.add()
        op_mul.name = name + '_MUL'
        op_mul.type = 'MUL'
        op_mul.input.extend([a])
        op_mul.input.extend([X])
        output_name_mul = X + '_xx'
        op_mul.output.extend([output_name_mul])
        op_mul.output_shape.extend(op.output_shape)
        #tensor data is variable,so didn't creat new tensor
        self._maceOpArray = np.append(self._maceOpArray,op_mul)

        #creat ADD
        op_add = self._SGSModel.op.add()
        op_add.name = name + '_ADD'
        op_add.type = 'ADD'
        op_add.input.extend([output_name_mul])
        op_add.input.extend([Y])
        op_add.output[:] = op.output[:]
        op_add.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_add)
        #remove axpy op
        self.remove_op_with_name(name)

    def split_BatchNorm(self, op):
        # yi = xi * scale_value + offset_value
        name = op.name
        xi = op.input[0]
        scale_value = op.input[1]
        offset_value = op.input[2]
        #creat Mul
        op_mul = self._SGSModel.op.add()
        op_mul.name = name + '_MUL'
        op_mul.type = 'MUL'
        op_mul.input.extend([xi])
        op_mul.input.extend([scale_value])
        if op.output[0].count('_xx') == 2:
          output_name_mul = xi + '_xx'
        else:
          output_name_mul = xi + '_MUL'
        op_mul.output.extend([output_name_mul])
        op_mul.output_shape.extend(op.output_shape)
        #tensor data is variable,so didn't creat new tensor
        self._maceOpArray = np.append(self._maceOpArray,op_mul)

        #creat ADD
        op_add = self._SGSModel.op.add()
        op_add.name = name + '_ADD'
        op_add.type = 'ADD'
        op_add.input.extend([output_name_mul])
        op_add.input.extend([offset_value])
        op_add.output[:] = op.output[:]
        op_add.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_add)
        #remove BatchNorm op
        self.remove_op_with_name(name)


    def split_Concat(self, op):
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        inputShape = self.get_shape_by_name(xi)
        if inputTensor.data_format == mace_pb2.DT_NHWC:
          for op_ in self._SGSModel.op:
             if op_ == op:
              op_.type = "CONCATENATION"
              self._maceOpArray = np.append(self._maceOpArray,op_)
              outputName = op_.output[0]
              outputTensor = self.find_tensor_by_name(outputName)
              outputTensor.data_format = mace_pb2.DT_NHWC
        else:
          for op_ in self._SGSModel.op:
             if op_ == op:
              axis = 1
              output_shape = op_.output_shape[:]
              op_.type = "CONCATENATION"
              arg = op_.arg
              for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_axis_str:
                  arg[i].i = len(inputShape) + arg[i].i if arg[i].i < 0 else arg[i].i
                  if len(output_shape[0].dims) == 4:#nchw -> nhwc
                    if arg[i].i == 1:
                      arg[i].i = 3
                    elif arg[i].i == 2:
                      arg[i].i = 1
                    elif arg[i].i == 3:
                      arg[i].i = 2
              self._maceOpArray = np.append(self._maceOpArray,op_)
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
              if group == 1:
                  op_.type = "CONV_2D"
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
                  axis_data = [3]
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

                #creat filter
                      filter_tensor_name = op_conv.name + "_filter#" + str(i)
                      filter_tensor_shape = copy.deepcopy(tensor_weight_shape)
                      filter_tensor_shape[0] = int(tensor_weight_shape[0] / group)
                      offset = len(tensor_weight.float_data) / group
                      filter_tesnor_value = tensor_weight.float_data[int(i*offset):int((i+1)*offset)]
                      self.add_tensor(self._SGSModel ,filter_tensor_name, filter_tensor_shape,
                              mace_pb2.DT_FLOAT, filter_tesnor_value)
                #creat bais
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

    def split_CReLU(self, op):
        # (mul * -1 + leakRelu)    leakRelu
        #           |              |
        #           | - - - -|- - -|
        #                    |
        #                  concat
        op_name = op.name
        xi = op.input[0]
        inputShape = self.get_shape_by_name(xi)
        [n,c,h,w] = self.get_shape_by_name(xi)
        slope,axis = 0.0,1
        arg = op.arg
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

    def split_ContinuationIndicator(self, op):
        op_name = op.name
        [t,b] = op.output_shape[0].dims[:]
        output_name = op.output[0]
        output_tensor = self.find_tensor_by_name(output_name)
        value = np.ones(t*b,dtype=np.float32).reshape(t,b)
        value[0,0] = 0.0
        output_tensor.data_type = mace_pb2.DT_FLOAT
        output_tensor.float_data.extend(list(value))

    def split_Crop(self, op):
        op_name = op.name
        input_data_name = op.input[0]
        [n,c,h,w] = op.output_shape[0].dims[:]
        arg = op.arg
        offset = []
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
        elif offset != []:
          for i in six.moves.range(len(offset)):
            offset_list[3-i] = offset[len(offset)-1-i]
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
    def split_Clip(self, op):
        for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "RELU6"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Deconv2D(self, op):
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        if inputTensor.data_format == mace_pb2.DT_NHWC:
          six.print_("Deconv2D input data format must be NCHW")
          assert 0
        arg = op.arg
        [ni,ci,hi,wi] = input_shape = self.get_shape_by_name(xi)
        [n,c,h,w] = output_shape = op.output_shape[0].dims[:]
        strideW,strideH = 0,0
        dilationW,dilationH = 1,1
        depthwise = False
        op_name = op.name
        input_data_name = op.input[0]
        filter_name = op.input[1]
        biase_name = op.input[2]
        filter_tensor = self.find_tensor_by_name(filter_name)
        [do,di,kh,kw] = filter_tensor.dims[:]
        filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(do,di,kh,kw)
        # reverse kernel
        filter_data = filter_data[:, :, ::-1, ::-1]
        filter_tensor.float_data[:] = filter_data.flat
        group = 0
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == MaceKeyword.mace_padding_values_str:
            paddingL,paddingR,paddingT,paddingB= arg[i].ints
          elif name == MaceKeyword.mace_strides_str:
            strideW,strideH = arg[i].ints
          elif name == MaceKeyword.mace_dilations_str:
            dilationW,dilationH = arg[i].ints
          elif name == 'group':
            group = arg[i].i
        if group == c and group != 1:
           depthwise = True
        elif group > 1 and group != c:
           mace_check(False,
                          "Sigmastar do not support group Deconverlution yet")

        def computeOutSidePadding(input_shape,output_shape,strideH,strideW,kernel_shape):
          [ni,ci,hi,wi] = input_shape
          [n,c,h,w] = output_shape
          input_pading_shape_H = hi + (hi - 1)*(strideH - 1)
          input_pading_shape_W = wi + (wi - 1)*(strideW - 1)
          round_func = math.floor
          pad_h = round_func((h - 1 - input_pading_shape_H + kernel_shape)/2)
          pad_w = round_func((w - 1 - input_pading_shape_W + kernel_shape)/2)
          dilation_output_h = input_pading_shape_H + 2 * pad_h
          dilation_output_w = input_pading_shape_W + 2 * pad_w
          return pad_h,pad_w,dilation_output_h,dilation_output_w

        pad_h,pad_w,dilation_output_h,dilation_output_w = computeOutSidePadding(input_shape,output_shape,strideH,strideW,filter_data.shape[2])
        #                #
        # creat dilation #
        #                #
        op_dilation = self._SGSModel.op.add()
        op_dilation.name = 'Dilation'
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
            op_dep_conv.input.extend([output_name_dilation])
            depth_filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(1,do,kh,kw)
            filter_tensor.float_data[:] = depth_filter_data.flat
            filter_tensor.dims[:] = [1,do,kh,kw]
            op_dep_conv.input.extend([filter_name])
            op_dep_conv.input.extend([biase_name])
            op_dep_conv.output[:] = op.output[:]
            op_dep_conv.output_shape.extend(op.output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_dep_conv)
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
            op_conv.input.extend([output_name_dilation])
            op_conv.input.extend([filter_name])
            op_conv.input.extend([biase_name])
            op_conv.output[:] = op.output[:]
            op_conv.output_shape.extend(op.output_shape)
            self._maceOpArray = np.append(self._maceOpArray,op_conv)
        self.remove_op_with_name(op_name)

    def split_DepthwiseConv2d(self, op):
        arg = op.arg
        [n,c,h,w] = op.output_shape[0].dims[:]

        op_name = op.name
        input_data_name = op.input[0]
        filter_name = op.input[1]
        biase_name = op.input[2]
        biase_tensor = self.find_tensor_by_name(biase_name)
        filter_tensor = self.find_tensor_by_name(filter_name)
        [do,di,kh,kw] = filter_tensor.dims[:]
        # if kh != 3 and kh < 9:#GDConv
        #   # 1.pad kernel
        #   filter_data = np.array(filter_tensor.float_data[:],dtype = np.float32).reshape(do,di,kh,kw)
        #   pad_value = int(9-kh)
        #   filter_data = np.pad(filter_data,((0,0),(0,0),(0,pad_value),(0,pad_value)),'constant',constant_values=(0,0))

        #   # 2.slice kernel
        #   filter_list = {}
        #   filter_num = 0
        #   for h in range(3):
        #     for w in range(3):
        #         end1 = int(3*(h+1))
        #         end2 = int(3*(w+1))
        #         slice_one = filter_data[:, :, 3*h:end1, 3*w:end2]
        #         filter_list[filter_num] = slice_one
        #         filter_num += 1

        #   # 3.pad inputs
        #   [ni,nc,nh,nw] = self.get_shape_by_name(input_data_name)
        #   op_pad = self._SGSModel.op.add()
        #   op_pad.name = input_data_name + '_pad'
        #   op_pad.type = 'PAD'
        #   shape_tensor_name = op_pad.name + '_shape'
        #   shape_tensor_data = [0,0,0,pad_value,0,pad_value,0,0]#[n,h,w,c]
        #   shape_tensor_shape = [4,2]
        #   self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
        #   mace_pb2.DT_INT32, shape_tensor_data)
        #   op_pad.input.extend([input_data_name])
        #   op_pad.input.extend([shape_tensor_name])
        #   output_op_pad = op_pad.name + '_output'
        #   op_pad.output.extend([output_op_pad])
        #   op_pad.output_shape.add()
        #   output_shape_pad = [ni,nc,int(nh + pad_value),int(nw + pad_value)]
        #   op_pad.output_shape[0].dims.extend(output_shape_pad)
        #   self._maceOpArray = np.append(self._maceOpArray,op_pad)

        #   # 4.slice inputs
        #   input_list = {}
        #   input_num = 0
        #   for h in range(3):
        #     for w in range(3):
        #         begin1 = int(3*h)
        #         begin2 = int(3*w)
        #         op_slice = self._SGSModel.op.add()
        #         op_slice.name = op_name + '_slice' + "#" + str(input_num)
        #         op_slice.type = 'SLICE'
        #         begin_tensor_name = op_slice.name + '_begin'
        #         begin_tensor_data = [0,begin1,begin2,0]
        #         begin_tensor_shape = [4]
        #         self.add_tensor(self._SGSModel ,begin_tensor_name, begin_tensor_shape,
        #             mace_pb2.DT_INT32, begin_tensor_data)
        #         size_tensor_name = op_slice.name + '_size'
        #         size_tensor_data = [ni,3,3,nc]
        #         size_tensor_shape = [4]
        #         self.add_tensor(self._SGSModel ,size_tensor_name, size_tensor_shape,
        #             mace_pb2.DT_INT32, size_tensor_data)
        #         op_slice.input.extend([output_op_pad])
        #         op_slice.input.extend([begin_tensor_name])
        #         op_slice.input.extend([size_tensor_name])
        #         output_op_slice = op_slice.name + '_output'
        #         op_slice.output.extend([output_op_slice])
        #         op_slice.output_shape.add()
        #         op_slice.output_shape[0].dims.extend([ni,nc,3,3])
        #         self._maceOpArray = np.append(self._maceOpArray,op_slice)
        #         #slice_one = filter_data[3*h:end1, 3*w:end2]
        #         input_list[input_num] = op_slice
        #         input_num += 1

        #   # 5.depthwise
        #   sub_depthwise_output_tensor_list = []
        #   #sub_depthwise_output_tensor_shape = []
        #   for i in six.moves.range(9):
        #     op_conv = self._SGSModel.op.add()
        #     op_conv.name = op_name + '_depthwise_conv' + '#' + str(i)
        #     op_conv.type = 'DEPTHWISE_CONV_2D'
        #     strides_arg = op_conv.arg.add()
        #     strides_arg.name = 'strides'
        #     strides_arg.ints.extend([1,1])
        #     padding_arg = op_conv.arg.add()
        #     padding_arg.name = 'padding_values'
        #     padding_arg.ints.extend([0,0,0,0])
        #     filter_tensor_name = op_conv.name + '_filter'
        #     filter_data = filter_list[i]
        #     filter_shape = filter_data.shape
        #     self.add_tensor(self._SGSModel, filter_tensor_name, filter_shape,
        #             mace_pb2.DT_FLOAT, filter_data.flatten().tolist())
        #     bias_data = biase_tensor
        #     bias_tensor_name = op_conv.name + '_bias'
        #     self.add_tensor(self._SGSModel, bias_tensor_name, bias_data.dims[:],
        #                     mace_pb2.DT_FLOAT,bias_data.float_data[:])
        #     op_conv.input.extend([input_list[i].output[0]])
        #     op_conv.input.extend([filter_tensor_name])
        #     op_conv.input.extend([bias_tensor_name])
        #     output_op_conv_name = op_conv.name + '_output'
        #     op_conv.output.extend([output_op_conv_name])
        #     op_conv.output_shape.add()
        #     op_conv.output_shape[0].dims.extend([ni,nc,1,1])
        #     sub_depthwise_output_tensor_list.append(output_op_conv_name)
        #     self._maceOpArray = np.append(self._maceOpArray,op_conv)

        #   # 6.concat + reshape
        #   op_concat = self._SGSModel.op.add()
        #   op_concat.name = op_name + '_concat'
        #   op_concat.type = 'CONCATENATION'
        #   axis_arg = op_concat.arg.add()
        #   axis_arg.name = 'axis'
        #   axis_arg.i = 2
        #   for concat_name in sub_depthwise_output_tensor_list:
        #     op_concat.input.extend([concat_name])
        #   output_op_concat = op_concat.name + '_output'
        #   op_concat.output.extend([output_op_concat])
        #   op_concat.output_shape.add()
        #   op_concat.output_shape[0].dims.extend([ni,nc,1,9])
        #   self._maceOpArray = np.append(self._maceOpArray,op_concat)
        #   op_reshape = self._SGSModel.op.add()
        #   op_reshape.name = op_name + '_reshape'
        #   op_reshape.type = 'RESHAPE'
        #   op_reshape.input.extend([output_op_concat])
        #   reshape_output_tensor_name = op_reshape.name + '_output_shape'
        #   reshape_output_tensor_data = [ni,3,3,nc]#n,h,w,c
        #   reshape_output_tensor_shape = [4]
        #   self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
        #                   mace_pb2.DT_INT32, reshape_output_tensor_data)
        #   op_reshape.input.extend([reshape_output_tensor_name])
        #   output_op_reshape = op_reshape.name + '_output'
        #   op_reshape.output.extend([output_op_reshape])
        #   op_reshape.output_shape.add()
        #   op_reshape.output_shape[0].dims.extend([ni,nc,3,3])#n,c,h,w
        #   self._maceOpArray = np.append(self._maceOpArray,op_reshape)

        #   # 7.depthwise
        #   op_conv = self._SGSModel.op.add()
        #   op_conv.name = op_name + '_depthwise_conv' + "_finale"
        #   op_conv.type = 'DEPTHWISE_CONV_2D'
        #   strides_arg = op_conv.arg.add()
        #   strides_arg.name = 'strides'
        #   strides_arg.ints.extend([1,1])
        #   padding_arg = op_conv.arg.add()
        #   padding_arg.name = 'padding_values'
        #   padding_arg.ints.extend([0,0,0,0])
        #   filter_tensor_name = op_conv.name + '_filter'
        #   filter_data = np.ones((1,nc,3,3))
        #   filter_shape = filter_data.shape
        #   self.add_tensor(self._SGSModel, filter_tensor_name, filter_shape,
        #           mace_pb2.DT_FLOAT, filter_data.flatten().tolist())
        #   bias_data = biase_tensor
        #   bias_tensor_name = op_conv.name + '_bias'
        #   self.add_tensor(self._SGSModel, bias_tensor_name, bias_data.dims[:],
        #                   mace_pb2.DT_FLOAT,bias_data.float_data[:])
        #   op_conv.input.extend([output_op_reshape])
        #   op_conv.input.extend([filter_tensor_name])
        #   op_conv.input.extend([bias_tensor_name])
        #   op_conv.output[:] = op.output[:]
        #   op_conv.output_shape.extend(op.output_shape)
        #   self._maceOpArray = np.append(self._maceOpArray,op_conv)

        #   self.remove_op_with_name(op_name)
        # else:
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "DEPTHWISE_CONV_2D"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Dropout(self, op):
        op_name = op.name
        xi = op.input[0]
        xi_shape = self.get_shape_by_name (xi)
        out_shape = op.output_shape[0].dims[:]
        op_output_shape = copy.deepcopy(out_shape)
        if len(xi_shape) == 4:
            tmp = out_shape[1]
            out_shape[1] = out_shape[3]
            out_shape[3] = tmp
            op_output_shape = out_shape
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
        x1 = op.input[0]
        inputTensor1 = self.find_tensor_by_name(x1)
        x1_format = inputTensor1.data_format
        isSub = False
        for inputName in op.input:
          input_data_format = self.find_tensor_by_name(inputName).data_format
          if input_data_format != x1_format:
            six.print_("Eltwise layer must operater in same data format")
            assert 0

        for op_ in self._SGSModel.op:
          if op_ == op:
            arg = op_.arg
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
                  six.print_("eltwise do not support")
                self._maceOpArray = np.append(self._maceOpArray, op_)
            if inputTensor1.data_format == mace_pb2.DT_NHWC:
              for outputName in op_.output:
                outputTensor = self.find_tensor_by_name(outputName)
                outputTensor.data_format = mace_pb2.DT_NHWC

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
           op_reshape.output_shape[0].dims.extend([n,1*c*h*w,1,1])#n,c,h,w
           self._maceOpArray = np.append(self._maceOpArray,op_reshape)

           #creat ori fullconnect conv
           op_conv = self._SGSModel.op.add()
           op_conv.name = op_name + '_fully_connet#1'
           op_conv.type = 'FULLY_CONNECTED'
           '''
           strides_arg = op_conv.arg.add()
           strides_arg.name = 'strides'
           strides_arg.ints.extend([1,1])
           padding_arg = op_conv.arg.add()
           padding_arg.name = 'padding_values'
           padding_arg.ints.extend([0,0,0,0])
           '''
           tensor_weight = self.find_tensor_by_name(weight)
           temp_data = np.array(tensor_weight.float_data).reshape(op.output_shape[0].dims[1],1,c,h,w)
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

    def split_LSTM(self, op):
        op_name = op.name
        xi = op.input[0]
        xi_shape = self.get_shape_by_name(xi)
        T = op.input[1]
        T_shape = self.get_shape_by_name(T)
        num_output = op.output_shape[0].dims[-1]
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
          op_reshape.output_shape[0].dims.extend([xi_shape[0],tmp,1,1])#n,c,h,w
          self._maceOpArray = np.append(self._maceOpArray,op_reshape)
          lstm_data_input_tensor = output_op_reshape

        name_prefix = "sgs_subnet_lstm"
        # add h0
        h0_name = name_prefix + '_h0'
        h0_data = np.zeros(num_output)
        h0_shape = [1,num_output,1,1]
        self.add_tensor(self._SGSModel, h0_name, h0_shape,
                        mace_pb2.DT_FLOAT, h0_data)
        # add c0
        c0_name = name_prefix + '_c0'
        c0_data = np.zeros(num_output)
        c0_shape = [1,num_output,1,1]
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
        input_output_map[name_prefix + '_output'] = op.output_shape[0].dims[:]
        np.save("./input_output_shape.npy",input_output_map, allow_pickle=True)
        #                #
        # creat SGS_LSTM #
        #                #
        op_SGS_LSTM = self._SGSModel.op.add()
        op_SGS_LSTM.name = 'SGS_LSTM'
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
        single_shape[1] = op.output_shape[0].dims[-1]
        for i in six.moves.range(T_time):
          tmp_out_name = op.name + '_output' + str(i)
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
        concat_shape[0] = op.output_shape[0].dims[0]
        concat_shape[1] = op.output_shape[0].dims[-1]
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
        op_reshape.output[:] = op.output[:]
        #op_reshape.output_shape.add()
        #op_reshape.output_shape[0].dims.extend([op.output_shape[0].dims[0],1,op.output_shape[0].dims[1]])#n,c,h,w
        op_reshape.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_reshape)
        self.remove_op_with_name(op_name)

    def split_Pooling(self, op):
        xi = op.input[0]
        inputTensor = self.find_tensor_by_name(xi)
        if inputTensor.data_format == mace_pb2.DT_NHWC:
          six.print_("Pooling input data format must be NCHW")
          assert 0
        kernel_h,kernel_w = -1,-1
        paddingL,paddingR,paddingT,paddingB = 0,0,0,0
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
                elif name == MaceKeyword.mace_padding_values_str:
                    paddingL,paddingR,paddingT,paddingB = arg[i].ints
                elif name == MaceKeyword.mace_global_pooling_str:
                    input_op_name = op_.input[0]
                    #find output_shape of bottom op
                    for input_op in self._SGSModel.op:
                        output_name_list = input_op.output
                        for output_name in output_name_list:
                          if output_name == input_op_name:
                            kernel_h,kernel_w = input_op.output_shape[0].dims[2:]
              for i in six.moves.range(len(arg)):
                name = arg[i].name
                if name == MaceKeyword.mace_kernel_str:
                    if kernel_h != -1:
                      arg[i].ints[:] = []
                      arg[i].ints.extend([kernel_h,kernel_w])
              if op_.type == 'AVERAGE_POOL_2D' and  paddingL != 0:
                [ni,ci,hi,wi] = self.get_shape_by_name(xi)
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
                output_shape_pad = [ni,ci,int(hi+2*paddingT),int(wi+2*paddingL)]
                op_pad.output_shape[0].dims.extend(output_shape_pad)
                self._maceOpArray = np.append(self._maceOpArray,op_pad)
                #modify average pooling
                arg = op_.arg
                for i in six.moves.range(len(arg)):
                  name = arg[i].name
                  if name == MaceKeyword.mace_padding_values_str:
                      arg[i].ints[:] = [0,0,0,0]
                op_.input[0] = output_op_pad
              self._maceOpArray = np.append(self._maceOpArray,op_)

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
        for name in op.input:
            for key,value in self._inputShapeMap.items():
                if key == name:
                    [img_height,img_width]= value[2:]
        num_prior = len(aspect_ratios) * len(min_sizes) + len(max_sizes)

        #gen anchor box
        top_data = SGSModel_genAnchors.genAnchors(layer_width,layer_height,img_width,img_height,
                                                  aspect_ratios,variance,offset,num_prior,min_sizes,
                                                  max_sizes,clip,step_h,step_w,img_h,img_w)
        #add tensor
        self.add_tensor(self._SGSModel ,output_name, output_shape,
                        mace_pb2.DT_FLOAT, top_data)
        #remove priorBox op
        self.remove_op_with_name(op_name)

    def split_PriorBox_RFC(self, op):
            #init param
            #pdb.set_trace()
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
            #pdb.set_trace()
            #gen anchor box
            top_data = SGSModel_prior_box.genAnchors(layer_width,layer_height,img_width,img_height,
                                                      aspect_ratios,variance,offset,num_prior,min_sizes,
                                                      max_sizes,clip,step_h,step_w,img_h,img_w)
            #add tensor
            self.add_tensor(self._SGSModel ,output_name, output_shape,
                            mace_pb2.DT_FLOAT, top_data)
            #remove priorBox op
            self.remove_op_with_name(op_name)
    def split_Reduce(self, op):
        pass

    def split_Reshape(self, op):
       xi = op.input[0]
       op_output_shape = op.output_shape[0].dims
       op_name = op.name
       inputTensor = self.find_tensor_by_name(xi)
       input_shape = self.get_shape_by_name(xi)
       shape_total = 1
       for shape in input_shape:
         shape_total *= shape
       if inputTensor.data_format == mace_pb2.DT_NHWC:
         for op_ in self._SGSModel.op:
           if op_ == op:
             #output_shape = op_.output_shape[0].dims[:]
             output_tensor_name = op_.name + '_output_shape'
             output_tensor_data = op_output_shape
             output_tensor_shape = [len(op_output_shape)]
             self.add_tensor(self._SGSModel, output_tensor_name, output_tensor_shape,
                             mace_pb2.DT_INT32, output_tensor_data,mace_pb2.DT_NHWC)
             op_.input.extend([output_tensor_name])
             op_.type = "RESHAPE"
             self._maceOpArray = np.append(self._maceOpArray,op_)
             for outputName in op_.output:
               outputTensor = self.find_tensor_by_name(outputName)
               outputTensor.data_format = mace_pb2.DT_NHWC
       elif len(input_shape) == 4 and (shape_total not in input_shape):
           # creat transpose for NHWC to NCHW in tflite model
           #output_shape = self.get_shape_by_name(xi)
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
           if len(op_output_shape) == 4:
               # creat ori reshape layer
                op_reshape = self._SGSModel.op.add()
                op_reshape.name = op_name + '_reshape'
                op_reshape.type = 'RESHAPE'
                reshape_output_tensor_name = op_reshape.name + '_output_shape'
                reshape_output_tensor_data = op_output_shape
                reshape_output_tensor_shape = [len(op_output_shape)]
                self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                                mace_pb2.DT_INT32, reshape_output_tensor_data)
                op_reshape.input.extend([output_op_transpose])
                op_reshape.input.extend([reshape_output_tensor_name])
                op_reshape_output_name = op_reshape.name + '_output'
                op_reshape.output.extend([op_reshape_output_name])
                op_reshape.output_shape.add()
                [rn,rc,rh,rw] = op_output_shape
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
               # creat ori reshape layer
               op_reshape = self._SGSModel.op.add()
               op_reshape.name = op_name + '_reshape'
               op_reshape.type = 'RESHAPE'
               reshape_output_tensor_name = op_reshape.name + '_output_shape'
               reshape_output_tensor_data = op_output_shape
               reshape_output_tensor_shape = [len(op_output_shape)]
               self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                               mace_pb2.DT_INT32, reshape_output_tensor_data)
               op_reshape.input.extend([output_op_transpose])
               op_reshape.input.extend([reshape_output_tensor_name])
               op_reshape.output[:] =  op.output[:]
               op_reshape.output_shape.extend(op.output_shape)
               self._maceOpArray = np.append(self._maceOpArray,op_reshape)
               for outputName in op.output:
                 outputTensor = self.find_tensor_by_name(outputName)
                 outputTensor.data_format = mace_pb2.DT_NHWC
           self.remove_op_with_name(op_name)
       elif len(op_output_shape) == 4 and (shape_total not in input_shape):
           '''
           For shuffle_channel:
       [ 0, 116, 64, 64 ]
               ↓
           Reshape -> [ 0, 2, -1, 64, 64 ]
           Permute -> "0 ", "2 ", "1 ", "3 ", "4 "
           Reshape -> [ 0, 116, 64, 64 ]
           '''
           # creat ori reshape layer
           op_reshape = self._SGSModel.op.add()
           op_reshape.name = op_name + '_reshape'
           op_reshape.type = 'RESHAPE'
           reshape_output_tensor_name = op_reshape.name + '_output_shape'
           reshape_output_tensor_data = op_output_shape
           reshape_output_tensor_shape = [len(op_output_shape)]
           self.add_tensor(self._SGSModel, reshape_output_tensor_name, reshape_output_tensor_shape,
                           mace_pb2.DT_INT32, reshape_output_tensor_data)
           op_reshape.input.extend([xi])
           op_reshape.input.extend([reshape_output_tensor_name])
           op_reshape_output_name = op_reshape.name + '_output'
           op_reshape.output.extend([op_reshape_output_name])
           op_reshape.output_shape.add()
           [rn,rc,rh,rw] = op_output_shape
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
           op_transpose.output[:] =  op.output[:]
           op_transpose.output_shape.extend(op.output_shape)
           self._maceOpArray = np.append(self._maceOpArray,op_transpose)
       else:
           for op_ in self._SGSModel.op:
             if op_ == op:
               output_shape = op_.output_shape[0].dims[:]
               output_tensor_name = op_.name + '_output_shape'
               output_tensor_data = output_shape
               output_tensor_shape = [len(output_shape)]
               self.add_tensor(self._SGSModel, output_tensor_name, output_tensor_shape,
                               mace_pb2.DT_INT32, output_tensor_data)
               op_.input.extend([output_tensor_name])
               op_.type = "RESHAPE"
               self._maceOpArray = np.append(self._maceOpArray,op_)
               for outputName in op_.output:
                 outputTensor = self.find_tensor_by_name(outputName)
                 outputTensor.data_format = mace_pb2.DT_NCHW

    def split_Slice(self, op):
        # caffe slice == tensorflow split
        xi = op.input[0]

        slice_point_enable = False
        axis = 0
        inputTensor = self.find_tensor_by_name(xi)
        inputShape = self.get_shape_by_name(xi)
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
                     axis = op_.arg[i].i
                     axis = len(inputShape) + axis if axis < 0 else axis
                     axis_tensor_data = [axis]
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
                for i in six.moves.range(len(arg)):
                  name = arg[i].name
                  if name == MaceKeyword.mace_axis_str:
                    axis = len(inputShape) + arg[i].i if arg[i].i < 0 else arg[i].i
                    arg[i].i = len(inputShape) + arg[i].i if arg[i].i < 0 else arg[i].i
                    if len(output_shape[0].dims) == 4:#nchw -> nhwc
                      if arg[i].i == 1:
                        arg[i].i = 3
                      elif arg[i].i == 2:
                        arg[i].i = 1
                      elif arg[i].i == 3:
                        arg[i].i = 2
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
                self._maceOpArray = np.append(self._maceOpArray, op_)

    def split_Split(self, op):
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
        for op_ in self._SGSModel.op:
          if op_ == op:
            op_.type = "SOFTMAX"
            self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Scale(self, op):
        op_name = op.name
        xi = op.input[0]
        scale = op.input[1]
        scale_shape = self.get_shape_by_name(scale)
        arg = op.arg
        output_shape = op.output_shape[0].dims
        axis = 1
        for i in six.moves.range(len(arg)):
           name = arg[i].name
           if name == MaceKeyword.mace_axis_str:
               axis = arg[i].i
        if len(op.input) == 3: #has bias
          bias = op.input[2]
          op_mul = self._SGSModel.op.add()
          op_mul.name = op_name + '_mul'
          op_mul.type = 'MUL'
          op_mul.input.extend([xi,scale])
          op_mul_output = op_mul.name + '_output'
          op_mul.output.extend([op_mul_output])
          op_mul.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_mul)
          op_add = self._SGSModel.op.add()
          op_add.name = name + '_add'
          op_add.type = 'ADD'
          op_add.input.extend([op_mul_output])
          op_add.input.extend([bias])
          op_add.output[:] = op.output[:]
          op_add.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_add)
        else:
          op_mul = self._SGSModel.op.add()
          op_mul.name = op_name + '_mul'
          op_mul.type = 'MUL'
          op_mul.input.extend([xi])
          op_mul.input.extend([scale])
          op_mul.output[:] =  op.output[:]
          op_mul.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_mul)
        self.remove_op_with_name(op_name)

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
          op_transpose_2 = self._SGSModel.op.add()
          op_transpose_2.name = op_name + '_transpose#1'
          op_transpose_2.type = 'TRANSPOSE'
          shape_tensor_name = op_transpose_2.name + '_shape'
          shape_tensor_data = [n,c,h,w]
          shape_tensor_shape = [4]
          self.add_tensor(self._SGSModel ,shape_tensor_name, shape_tensor_shape,
              mace_pb2.DT_INT32, shape_tensor_data)
          op_transpose_2.input.extend([output_op_transpose])
          op_transpose_2.input.extend([shape_tensor_name])
          op_transpose_2.output.extend(op.output)
          op_transpose_2.output_shape.extend(op.output_shape)
          self._maceOpArray = np.append(self._maceOpArray,op_transpose_2)
          self.remove_op_with_name(op_name)

          # modify outputTensor data_format
          for outputName in op.output:
             outputTensor = self.find_tensor_by_name(outputName)
             outputTensor.data_format = mace_pb2.DT_NHWC
        else:
          for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "TRANSPOSE"
              self._maceOpArray = np.append(self._maceOpArray,op_)

    def split_Tile(self, op):
        op_name = op.name
        xi = op.input[0]
        arg = op.arg
        output_shape = op.output_shape[0].dims
        op_input = op.input[0]
        for i in six.moves.range(len(arg)):
           name = arg[i].name
           if name == MaceKeyword.mace_axis_str:
             if len(output_shape) == 4:#nchw -> nhwc
               if arg[i].i == 1:
                 axis = 3
               elif arg[i].i == 2:
                 axis = 1
               elif arg[i].i == 3:
                 axis = 2
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
        for op_ in self._SGSModel.op:
            if op_ == op:
              op_.type = "CUSTOM"
              op_.name = op.name + "_SGS_ROIPooling"
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
        op_transpose.output_shape.add()
        op_transpose.output_shape[0].dims.extend([n,int(4*c),h,w])
        self._maceOpArray = np.append(self._maceOpArray,op_transpose)

        #remove op
        self.remove_op_with_name(op_name)

    def split_Reverse(self, op):
        [ni,ci,hi,wi] = self.get_shape_by_name(op.input[0])
        op_name = op.name
        xi = op.input[0]#NHWC in interpreter

        # creat STRIDED_SLICE
        op_stride_slice = self._SGSModel.op.add()
        op_stride_slice.name = op_name + '_stride_slice'
        op_stride_slice.type = 'STRIDED_SLICE'
        begin_tensor_name = op_stride_slice.name + '_begin'
        begin_tensor_data = [0,0,0,-1]
        begin_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,begin_tensor_name, begin_tensor_shape,
            mace_pb2.DT_INT32, begin_tensor_data)
        end_tensor_name = op_stride_slice.name + '_end'
        end_tensor_data = [ni+1,hi+1,wi+1,-(ci+1)]
        end_tensor_shape = [4]
        self.add_tensor(self._SGSModel ,end_tensor_name, end_tensor_shape,
            mace_pb2.DT_INT32, end_tensor_data)
        stride_tensor_name = op_stride_slice.name + '_stride'
        stride_tensor_data = [1,1,1,-1]
        stride_tensor_shape = [4]
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

    def split_Power(self, op):
        # yi = xi * scale_value + offset_value
        op_name = op.name
        xi = op.input[0]
        arg = op.arg
        for i in six.moves.range(len(arg)):
          name = arg[i].name
          if name == 'power':
            power = arg[i].f

        scale_value = op.input[1]
        offset_value = op.input[2]
        mace_check(power % 1 == 0 and power > 0,
                       "power value not support yet.")
        power = int(power)
        #creat Mul
        op_mul = self._SGSModel.op.add()
        op_mul.name = op_name + '_MUL'
        op_mul.type = 'MUL'
        op_mul.input.extend([xi])
        op_mul.input.extend([scale_value])
        output_name_mul = op_mul.name + '_output'
        op_mul.output.extend([output_name_mul])
        op_mul.output_shape.extend(op.output_shape)
        #tensor data is variable,so didn't creat new tensor
        self._maceOpArray = np.append(self._maceOpArray,op_mul)

        #creat ADD
        op_add = self._SGSModel.op.add()
        op_add.name = op_name + '_ADD'
        op_add.type = 'ADD'
        op_add.input.extend([output_name_mul])
        op_add.input.extend([offset_value])
        if power == 1:
         op_add.output[:] = op.output[:]
        else:
         output_name_add = op_add.name + '_output'
         op_add.output.extend([output_name_add])
        op_add.output_shape.extend(op.output_shape)
        self._maceOpArray = np.append(self._maceOpArray,op_add)

        loop_num = power - 1
        for i in six.moves.range(loop_num):
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
            op_mul.output[:] = op.output[:]
          else:
            output_name_mul = op_mul.name + '_output#' + str(i)
            op_mul.output.extend([output_name_mul])
          op_mul.output_shape.extend(op.output_shape)
          #tensor data is variable,so didn't creat new tensor
          self._maceOpArray = np.append(self._maceOpArray,op_mul)

        #remove BatchNorm op
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
