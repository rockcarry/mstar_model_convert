import numpy as np
from TFLitePostProcess import *
from  anchor_param import *
from third_party import tflite
import math
import pdb

def buildGraph(sgs_builder,model_config):
    """

    :return:
    """
    """
        model_config = {"name":"SGS_LSTM",
          "input" : [input_name,indicator,h_0,c_0,],
          "input_shape" : [input_shape,indicator_shape,h0_shape,c0_shape],
          "out_shapes" : [output_shape]}
    """
    DEBUG = False#
    num_output = 128
    input_shape = model_config["input_shape"][0]
    indicator_shape = model_config["input_shape"][1]
    h0_shape = model_config["input_shape"][2]
    c0_shape = model_config["input_shape"][3]

    sgs_builder.buildBuffer('NULL')#the 0th entry of this array must be an empty buffer (sentinel).
    sgs_builder.buildTensor(model_config["input_shape"][0],model_config["input"][0])
    sgs_builder.buildTensor(model_config["input_shape"][1],model_config["input"][1])
    sgs_builder.buildTensor(model_config["input_shape"][2],model_config["input"][2])
    sgs_builder.buildTensor(model_config["input_shape"][3],model_config["input"][3])

    ## const 1 tensor
    constant_data = [1]
    constant_shape = [1]
    constant_data_vector = []
    for value in constant_data:
        constant_data_vector += bytearray(struct.pack("f", value))
    sgs_builder.buildBuffer("constant_data",constant_data_vector)
    sgs_builder.buildTensor(constant_shape,"mul_constant",sgs_builder.getBufferByName("constant_data"),tflite.TensorType.TensorType().FLOAT32)

    #                                   #
    #   creat reshape_input * 1         #
    #                                   #
    mul_indicator_input_arrays = []
    mul_indicator_input_arrays.append(model_config["input"][0])
    mul_indicator_input_arrays.append("mul_constant")
    sgs_builder.buildTensor(input_shape,"mul_constant_input#output")
    mul_indicator_output_arrays = []
    mul_indicator_output_arrays.append('mul_constant_input#output')
    sgs_builder.buildOperatorCode("mul_constant_input",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("mul_constant_input",mul_indicator_input_arrays,mul_indicator_output_arrays,tflite.BuiltinOptions.BuiltinOptions().MulOptions,None)



    #                 #
    #  h0 * indicator #
    #                 #
    mul_indicator_h_0_input_arrays = []
    mul_indicator_h_0_input_arrays.append(model_config["input"][2])
    mul_indicator_h_0_input_arrays.append(model_config["input"][1])
    sgs_builder.buildTensor(model_config["input_shape"][2],"mul_indicator_h_0#output")
    mul_indicator_h_0_output_arrays = []
    mul_indicator_h_0_output_arrays.append('mul_indicator_h_0#output')
    sgs_builder.buildOperatorCode("mul_indicator_h_0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("mul_indicator_h_0",mul_indicator_h_0_input_arrays,mul_indicator_h_0_output_arrays,tflite.BuiltinOptions.BuiltinOptions().MulOptions,None)
    #              #
    # creat concat #
    #              #
    concat_in_tensors = []
    concat_in_tensors.append('mul_constant_input#output')
    concat_in_tensors.append("mul_indicator_h_0#output")
    concat_out_tensors = []
    sgs_builder.buildTensor([1,1,1,h0_shape[-1] + input_shape[-1]],"concat_out")
    concat_out_tensors.append("concat_out")
    sgs_builder.buildOperatorCode("concat_1",tflite.BuiltinOperator.BuiltinOperator().CONCATENATION)
    concat_optionts = sgs_builder.createConcatenationOptions(3, 0)
    sgs_builder.buildOperator("concat_1",concat_in_tensors,concat_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions,concat_optionts)
    #             #
    #creat CONV2D #
    #             #
    weight_bias_load = np.load("./weight_biase_data.npz",allow_pickle=True)
    weight_data = weight_bias_load['weight']
    bias_data = weight_bias_load['bias']
    #creat weight buffer and tensor
    data = np.array(weight_data)
    data = data.reshape(weight_data.shape)
    data = data.transpose(0,2,3,1)
    data_shape = data.shape
    data = list(data.flat)
    if DEBUG:
      data = np.ones(2)
      bias_data = np.ones(2)
    weight_vector=[]
    for value in data:
        weight_vector += bytearray(struct.pack("f", value))
    sgs_builder.buildBuffer("weight_vector",weight_vector)
    sgs_builder.buildTensor(data_shape,"weight",sgs_builder.getBufferByName("weight_vector"),tflite.TensorType.TensorType().FLOAT32)
    #creat bais buffer and tensor
    bias_vector=[]
    for value in bias_data:
        bias_vector += bytearray(struct.pack("f", value))
    sgs_builder.buildBuffer("bias_vector",bias_vector)
    sgs_builder.buildTensor(bias_data.shape,"bias",sgs_builder.getBufferByName("bias_vector"),tflite.TensorType.TensorType().FLOAT32)
    conv_in_tensors = []
    conv_in_tensors.append("concat_out")
    conv_in_tensors.append("weight")
    conv_in_tensors.append("bias")
    conv_out_tensors = []
    conv_output_shape_0_dim = data_shape[0]
    sgs_builder.buildTensor([1,1,1,conv_output_shape_0_dim],"conv2d_out")
    conv_out_tensors.append("conv2d_out")
    sgs_builder.buildOperatorCode("Conv2d_1",tflite.BuiltinOperator.BuiltinOperator().CONV_2D)
    #createConv2DOptions(self,padding,strideW,strideH,fusedActivationFunction,dilationWFactor,dilationHFactor,paddingLeft,paddingRight,paddingTop,paddingBottom):
    Conv2DOptions = sgs_builder.createConv2DOptions(0,1,1,0,1,1,0,0,0,0) #padding 2 is caffe
    sgs_builder.buildOperator("Conv2d_1",conv_in_tensors,conv_out_tensors,tflite.BuiltinOptions.BuiltinOptions().Conv2DOptions,Conv2DOptions)

    #             #
    # creat split #
    #             #
    general_shape = [1,1,1,conv_output_shape_0_dim//4]
    ## axis tensor
    axis_data = [3]
    axis_shape = [1]
    axis_data_vector = []
    for value in axis_data:
        axis_data_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("axis_data",axis_data_vector)
    sgs_builder.buildTensor(axis_shape,"split_axis",sgs_builder.getBufferByName("axis_data"),tflite.TensorType.TensorType().INT32)
    split_input_arrays = []
    split_input_arrays.append("split_axis")
    split_input_arrays.append("conv2d_out")
    split_out_arrays = []
    sgs_builder.buildTensor(general_shape,"split_1_output#Tu")
    sgs_builder.buildTensor(general_shape,"split_1_output#Tf")
    sgs_builder.buildTensor(general_shape,"split_1_output#To")
    sgs_builder.buildTensor(general_shape,"split_1_output#Tg")
    '''
    split_out_arrays.append("split_1_output#Tu")
    split_out_arrays.append("split_1_output#Tf")
    split_out_arrays.append("split_1_output#To")
    split_out_arrays.append("split_1_output#Tg")
    '''
    #for onnx
    split_out_arrays.append("split_1_output#Tu")
    split_out_arrays.append("split_1_output#To")
    split_out_arrays.append("split_1_output#Tf")
    split_out_arrays.append("split_1_output#Tg")
    sgs_builder.buildOperatorCode("split_1",tflite.BuiltinOperator.BuiltinOperator().SPLIT)
    SplitOptions = sgs_builder.creatSplitOptions(len(split_out_arrays))
    sgs_builder.buildOperator("split_1",split_input_arrays,split_out_arrays,tflite.BuiltinOptions.BuiltinOptions().SplitOptions,SplitOptions)

    #                       #
    # creat LOGISTIC for Tu #
    #                       #
    logistic_Tu_input_arrays = []
    logistic_Tu_input_arrays.append("split_1_output#Tu")
    sgs_builder.buildTensor(general_shape,"split_1_output#Tu_output")
    logistic_Tu_output_arrays = []
    logistic_Tu_output_arrays.append("split_1_output#Tu_output")
    sgs_builder.buildOperatorCode("logistic_Tu",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("logistic_Tu",logistic_Tu_input_arrays,logistic_Tu_output_arrays,None,None)

    #                       #
    # creat LOGISTIC for Tf #
    #                       #
    logistic_Tf_input_arrays = []
    logistic_Tf_input_arrays.append("split_1_output#Tf")
    sgs_builder.buildTensor(general_shape,"split_1_output#Tf_output")
    logistic_Tf_output_arrays = []
    logistic_Tf_output_arrays.append("split_1_output#Tf_output")
    sgs_builder.buildOperatorCode("logistic_Tf",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("logistic_Tf",logistic_Tf_input_arrays,logistic_Tf_output_arrays,None,None)

    #                       #
    # creat LOGISTIC for To #
    #                       #
    logistic_To_input_arrays = []
    logistic_To_input_arrays.append("split_1_output#To")
    sgs_builder.buildTensor(general_shape,"split_1_output#To_output")
    logistic_To_output_arrays = []
    logistic_To_output_arrays.append("split_1_output#To_output")
    sgs_builder.buildOperatorCode("logistic_To",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("logistic_To",logistic_To_input_arrays,logistic_To_output_arrays,None,None)

    #                       #
    # creat Tanh for Tg     #
    #                       #
    tanh_Tg_input_arrays = []
    tanh_Tg_input_arrays.append("split_1_output#Tg")
    sgs_builder.buildTensor(general_shape,"split_1_output#Tg_output")
    tanh_Tg_output_arrays = []
    tanh_Tg_output_arrays.append("split_1_output#Tg_output")
    sgs_builder.buildOperatorCode("tanh_Tg",tflite.BuiltinOperator.BuiltinOperator().TANH)
    sgs_builder.buildOperator("tanh_Tg",tanh_Tg_input_arrays,tanh_Tg_output_arrays,None,None)

    #                #
    # creat Tf * c0  #
    #                #
    mul_Tf_c0_input_arrays = []
    mul_Tf_c0_input_arrays.append("split_1_output#Tf_output")
    mul_Tf_c0_input_arrays.append(model_config["input"][3])
    sgs_builder.buildTensor(general_shape,"mul_Tf_c0#output")
    mul_Tf_c0_output_arrays = []
    mul_Tf_c0_output_arrays.append('mul_Tf_c0#output')
    sgs_builder.buildOperatorCode("mul_Tf_c0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("mul_Tf_c0",mul_Tf_c0_input_arrays,mul_Tf_c0_output_arrays,tflite.BuiltinOptions.BuiltinOptions().MulOptions,None)

    #                #
    # creat Tu * Tg  #
    #                #
    mul_Tf_Tg_input_arrays = []
    mul_Tf_Tg_input_arrays.append("split_1_output#Tu_output")
    mul_Tf_Tg_input_arrays.append("split_1_output#Tg_output")
    sgs_builder.buildTensor(general_shape,"mul_Tf_Tg#output")
    mul_Tf_Tg_output_arrays = []
    mul_Tf_Tg_output_arrays.append('mul_Tf_Tg#output')
    sgs_builder.buildOperatorCode("mul_Tf_Tg",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("mul_Tf_Tg",mul_Tf_Tg_input_arrays,mul_Tf_Tg_output_arrays,tflite.BuiltinOptions.BuiltinOptions().MulOptions,None)


    #                                       #
    # creat add  Tf * c0 +  Tu * Tg = add_3 #
    #                                       #
    add_3_input_arrays = []
    add_3_input_arrays.append("mul_Tf_c0#output")
    add_3_input_arrays.append("mul_Tf_Tg#output")
    sgs_builder.buildTensor(general_shape,"add_3#output")
    add_3_output_arrays = []
    add_3_output_arrays.append('add_3#output') # C_n
    sgs_builder.buildOperatorCode("add_3",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("add_3",add_3_input_arrays,add_3_output_arrays,tflite.BuiltinOptions.BuiltinOptions().AddOptions,None)

    #                                        #
    #   creat add_3 * 1  = tang_add_3        #
    #                                        #
    mul_add3_output_input_arrays = []
    mul_add3_output_input_arrays.append("add_3#output")
    mul_add3_output_input_arrays.append("mul_constant")
    sgs_builder.buildTensor(general_shape,"mul_constant_add#output")
    mul_add3_output_output_arrays = []
    mul_add3_output_output_arrays.append('mul_constant_add#output')
    sgs_builder.buildOperatorCode("mul_constant_add",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("mul_constant_add",mul_add3_output_input_arrays,mul_add3_output_output_arrays,tflite.BuiltinOptions.BuiltinOptions().MulOptions,None)

    #                                        #
    #   creat Tanh for add_3  = tang_add_3   #
    #                                        #
    tanh_add_3_input_arrays = []
    tanh_add_3_input_arrays.append("mul_constant_add#output")
    sgs_builder.buildTensor(general_shape,"tanh_add_3#output")
    tanh_add_3_output_arrays = []
    tanh_add_3_output_arrays.append("tanh_add_3#output")
    sgs_builder.buildOperatorCode("tanh_add_3",tflite.BuiltinOperator.BuiltinOperator().TANH)
    sgs_builder.buildOperator("tanh_add_3",tanh_add_3_input_arrays,tanh_add_3_output_arrays,None,None)




    #                        #
    # creat To * tang_add_3  #
    #                        #
    mul_To_tang_add_3_input_arrays = []
    mul_To_tang_add_3_input_arrays.append("split_1_output#To_output")
    mul_To_tang_add_3_input_arrays.append("tanh_add_3#output")
    sgs_builder.buildTensor(general_shape,"h_n")
    mul_To_tang_add_3_output_arrays = []
    mul_To_tang_add_3_output_arrays.append('h_n')
    sgs_builder.buildOperatorCode("mul_To_add_3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("mul_To_add_3",mul_To_tang_add_3_input_arrays,mul_To_tang_add_3_output_arrays,tflite.BuiltinOptions.BuiltinOptions().MulOptions,None)

    #creat SubGraph and Model
    sgs_builder.subgraphs.append(sgs_builder.buildSubGraph(model_config["input"],mul_To_tang_add_3_output_arrays,model_config["name"]))
    sgs_builder.model = sgs_builder.createModel(3,sgs_builder.operator_codes,sgs_builder.subgraphs,model_config["name"],sgs_builder.buffers)
    file_identifier = b'TFL3'
    sgs_builder.builder.Finish(sgs_builder.model, file_identifier)
    buf = sgs_builder.builder.Output()
    return buf



def get_postprocess():
    #load file to parse data
    #1  input&output name and shapes
    shape = np.load("./input_output_shape.npy",allow_pickle=True)
    for key in shape.item():
        if key.split("_")[-1] == 'input':
          input_name = key
          input_shape = shape.item()[key]
        elif key.split("_")[-1] == 'h0':
          h_0 = key
          h0_shape = shape.item()[key]
        elif key.split("_")[-1] == 'c0':
          c_0 = key
          c0_shape = shape.item()[key]
        elif key.split("_")[-1] == 'output':
          output_name = key
          output_shape = shape.item()[key]
        elif key.split("_")[-1] == 'time':
          indicator = key
          indicator_shape = shape.item()[key]


    model_config = {"name":"SGS_LSTM_sub1",
          "input" : [input_name,indicator,h_0,c_0,],
          "input_shape" : [input_shape,indicator_shape,h0_shape,c0_shape],
          "out_shapes" : [output_shape]}

    LSTM = TFLitePostProcess()
    lSTM_buf = buildGraph(LSTM, model_config)
    outfilename = model_config["name"] + "_unroll.sim"
    with open(outfilename, 'wb') as f:
        f.write(lSTM_buf)
        f.close()
    print("\nWell Done! " + outfilename  + " generated!\n")

def model_postprocess():
    return get_postprocess()


