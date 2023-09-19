
import math
import os
import pdb
import numpy as np
import sys
import six

from calibrator_custom import sgs_chalk

def buildGraph(model_config,subid,total):
    """
    :return:
    """
    """
        model_config = {"name":"SGS_GRU",
          "input" : [input_name,indicator,h_0],
          "input_shape" : [input_shape,indicator_shape,h0_shape],
          "out_shapes" : [output_shape]}
    """
    DEBUG = False#
    #subid = lstm_num - 1 - i
    #total = lstm_num
    prefix = 'sub' + str(subid) + '_'
    num_output = 128
    input_shape = model_config["input_shape"][0]
    indicator_shape = model_config["input_shape"][1]
    h0_shape = model_config["input_shape"][2]
    #c0_shape = model_config["input_shape"][3]

    in0 = sgs_chalk.Input(model_config["input_shape"][0], name = model_config["input"][0])
    in1 = sgs_chalk.Input(model_config["input_shape"][1], name = model_config["input"][1])
    #in1 = sgs_chalk.Input((1,), name = model_config["input"][1])
    in2 = sgs_chalk.Input(model_config["input_shape"][2], name = model_config["input"][2])
    #in3 = sgs_chalk.Input(model_config["input_shape"][3], name = model_config["input"][3])

    #creat reshape_input * 1#
    constant1 = sgs_chalk.Tensor(data=np.array([1]).astype(np.float32),name = prefix + "mul_constant")
    mul_1 = sgs_chalk.Mul(in0, constant1, name = prefix + 'mul_constant_input#output')
    #  h0 * indicator #
    mul_2 = sgs_chalk.Mul(in2, in1, name = prefix + 'mul_indicator_h_0#output')
    # creat concat #
    concat_1 = sgs_chalk.Concatenation([mul_1,mul_2],axis=3,name=prefix + "concat_out")
    #creat CONV2D #
    file_name = './gru_data/weight_biase_data#' + str(total - subid - 1) + '.npz'
    weight_bias_load = np.load(file_name,allow_pickle=True)
    weight_data = weight_bias_load['weight']
    bias_data = weight_bias_load['bias']
    #creat weight buffer and tensor
    data = np.array(weight_data)
    data = data.reshape(weight_data.shape)
    data = data.transpose(0,2,3,1)
    data_shape = data.shape
    if DEBUG:
      data = np.ones(2)
      bias_data = np.ones(2)
    weight_tensor  =  sgs_chalk.Tensor(data=data.astype(np.float32),name = prefix + "weight")
    bias_tensor = sgs_chalk.Tensor(data=bias_data.astype(np.float32),name = prefix + "bias")
    conv2d_1 = sgs_chalk.Conv2d(concat_1,weight_tensor,bias_tensor,name=prefix + "conv2d_out")
    # creat split #
    output_list = sgs_chalk.Split(conv2d_1,NumSplits=2, axis=3,name=[prefix + "split_1_output#Tr",
                                  prefix + "split_1_output#Tz"])

    # creat LOGISTIC for Tr #
    logistic_Tr = sgs_chalk.Logistic(output_list[1],name=prefix + "split_1_output#Tr_output")
    # creat LOGISTIC for Tz #
    logistic_Tz = sgs_chalk.Logistic(output_list[0],name=prefix + "split_1_output#Tz_output")

    #creat CONV2D #
    file_name = './gru_data/weight_biase_data#' + str(total - subid - 1) + '.npz'
    weight_bias_load = np.load(file_name,allow_pickle=True)
    weight_data = weight_bias_load['Win']
    bias_data = weight_bias_load['bias_in']
    #creat weight buffer and tensor
    data = np.array(weight_data)
    data = data.reshape(weight_data.shape)
    data = data.transpose(0,2,3,1)
    data_shape = data.shape
    if DEBUG:
      data = np.ones(2)
      bias_data = np.ones(2)
    weight_in_tensor  =  sgs_chalk.Tensor(data=data.astype(np.float32),name = prefix + "weight_in")
    bias_in_tensor = sgs_chalk.Tensor(data=bias_data.astype(np.float32),name = prefix + "bias_in")
    conv2d_in = sgs_chalk.Conv2d(mul_1,weight_in_tensor,bias_in_tensor,name=prefix + "conv2d_in_out")


    #creat CONV2D #
    file_name = './gru_data/weight_biase_data#' + str(total - subid - 1) + '.npz'
    weight_bias_load = np.load(file_name,allow_pickle=True)
    weight_data = weight_bias_load['Whn']
    bias_data = weight_bias_load['bias_hn']
    #creat weight buffer and tensor
    data = np.array(weight_data)
    data = data.reshape(weight_data.shape)
    data = data.transpose(0,2,3,1)
    data_shape = data.shape
    if DEBUG:
      data = np.ones(2)
      bias_data = np.ones(2)
    weight_hn_tensor  =  sgs_chalk.Tensor(data=data.astype(np.float32),name = prefix + "weight_hn")
    bias_hn_tensor = sgs_chalk.Tensor(data=bias_data.astype(np.float32),name = prefix + "bias_hn")
    conv2d_hn = sgs_chalk.Conv2d(mul_2,weight_hn_tensor,bias_hn_tensor,name=prefix + "conv2d_hn_out")

    mul_3 = sgs_chalk.Mul(logistic_Tr, conv2d_hn, name = prefix + 'mul_logistic_Tr_conv2d_hn#output')
    add_1 = sgs_chalk.Add(conv2d_in,mul_3,name = prefix +'mul_logistic_Tr_conv2d_hn_Add_conv2d_in')
    tanh_Tn = sgs_chalk.Tanh(add_1,name = prefix + 'tanh_Tn')
    sub_1 = sgs_chalk.Sub(logistic_Tz,constant1,name = prefix + '1_sub_logistic_Tz')
    constant_1 = sgs_chalk.Tensor(data=np.array([-1]).astype(np.float32),name = prefix + "mul_constant_1")
    mul_4 = sgs_chalk.Mul(sub_1,constant_1,name=prefix + '1_sub_logistic_Tz_mul_mul_constant_1')
    mul_5 = sgs_chalk.Mul(mul_4,tanh_Tn,name = prefix + 'mul_constant_1_mul_tanh_Tn')
    mul_6 = sgs_chalk.Mul(logistic_Tz,mul_2,name= prefix + 'logistic_Tz#mul_2')
    out1 = sgs_chalk.Add(mul_5, mul_6, name = prefix + "h_n")

    outfilename = model_config["name"] + "_unroll.sim"
    model = sgs_chalk.Model([in0,in1,in2],out1,model_config["name"])
    model.save(outfilename)

    print("\nWell Done! " + outfilename  + " generated!\n")
    return outfilename


def get_postprocess():

    #load file to parse data
    path = './gru_data'
    files = os.listdir(path)
    files_num = len(files)
    gru_num = files_num//2
    outputfile = []
    #1  input&output name and shapes
    for i in six.moves.range(gru_num):
        file_name = './gru_data/input_output_shape#' + str(i) + '.npy'
        shape = np.load(file_name,allow_pickle=True)
        prefix = 'sub' + str(gru_num - 1 - i) + '_'
        for key in shape.item():
            if key.split("_")[-1] == 'input':
              input_name = prefix + 'gru_input'
              input_shape = shape.item()[key]
            elif key.split("_")[-1] == 'h0':
              h_0 = prefix + 'gru_h0'
              h0_shape = shape.item()[key]
            # elif key.split("_")[-1] == 'c0':
            #   c_0 = prefix + 'lstm_c0'
            #   c0_shape = shape.item()[key]
            elif key.split("_")[-1] == 'output':
              output_name = prefix + 'gru_output'
              output_shape = shape.item()[key]
            elif key.split("_")[-1] == 'time':
              indicator = prefix + 'gru_time'
              indicator_shape = shape.item()[key]

        modle_name = 'SGS_GRU_sub' + str(gru_num - 1 - i)
        model_config = {"name":modle_name,
              "input" : [input_name,indicator,h_0],
              "input_shape" : [input_shape,indicator_shape,h0_shape],
              "out_shapes" : [output_shape]}
        outfilename = buildGraph(model_config,gru_num - 1 - i,gru_num)
        outputfile.append(outfilename)
    return outputfile

def model_postprocess():
    return get_postprocess()
