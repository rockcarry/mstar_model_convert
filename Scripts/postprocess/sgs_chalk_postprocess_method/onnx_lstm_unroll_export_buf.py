
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
        model_config = {"name":"SGS_LSTM",
          "input" : [input_name,indicator,h_0,c_0,],
          "input_shape" : [input_shape,indicator_shape,h0_shape,c0_shape],
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
    c0_shape = model_config["input_shape"][3]

    in0 = sgs_chalk.Input(model_config["input_shape"][0], name = model_config["input"][0])
    in1 = sgs_chalk.Input(model_config["input_shape"][1], name = model_config["input"][1])
    #in1 = sgs_chalk.Input((1,), name = model_config["input"][1])
    in2 = sgs_chalk.Input(model_config["input_shape"][2], name = model_config["input"][2])
    in3 = sgs_chalk.Input(model_config["input_shape"][3], name = model_config["input"][3])

    #creat reshape_input * 1#
    constant1 = sgs_chalk.Tensor(data=np.array([1]).astype(np.float32),name = prefix + "mul_constant")
    mul_1 = sgs_chalk.Mul(in0, constant1, name = prefix + 'mul_constant_input#output')
    #  h0 * indicator #
    mul_2 = sgs_chalk.Mul(in2, in1, name = prefix + 'mul_indicator_h_0#output')
    # creat concat #
    concat_1 = sgs_chalk.Concatenation([mul_1,mul_2],axis=3,name=prefix + "concat_out")
    #creat CONV2D #
    file_name = './lstm_data/weight_biase_data#' + str(total - subid - 1) + '.npz'
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
    output_list = sgs_chalk.Split(conv2d_1,NumSplits=4, axis=3,name=[prefix + "split_1_output#Tu",
                                  prefix + "split_1_output#To",prefix + "split_1_output#Tf",prefix + "split_1_output#Tg"])
    # creat LOGISTIC for Tu #
    logistic_Tu = sgs_chalk.Logistic(output_list[0],name=prefix + "split_1_output#Tu_output")
    # creat LOGISTIC for Tf #
    logistic_Tf = sgs_chalk.Logistic(output_list[2],name=prefix + "split_1_output#Tf_output")
    # creat LOGISTIC for To #
    logistic_To = sgs_chalk.Logistic(output_list[1],name=prefix + "split_1_output#To_output")
    # creat Tanh for Tg     #
    tanh_Tg = sgs_chalk.Tanh(output_list[3],name=prefix + "split_1_output#Tg_output")
    # creat Tf * indicator  #
    mul_3 = sgs_chalk.Mul(logistic_Tf, in1, name = prefix + "mul_Tf_indicator#output")
    # creat Tf * c0  #
    mul_4 = sgs_chalk.Mul(mul_3, in3, name = prefix + "mul_Tf_c0#output")
    # creat Tu * Tg  #
    mul_5 = sgs_chalk.Mul(logistic_Tu, tanh_Tg, name = prefix + "mul_Tf_Tg#output")
    # creat add  Tf * c0 +  Tu * Tg = add_3 #
    add_1 = sgs_chalk.Add(mul_4, mul_5, name = prefix + "add_3#output")# C_n
    #   creat add_3 * 1  = tang_add_3        #
    mul_6 = sgs_chalk.Mul(add_1, constant1, name = prefix + "mul_constant_add#output")
    #   creat Tanh for add_3  = tang_add_3   #
    tanh_add_3 = sgs_chalk.Tanh(mul_6,name=prefix + "tanh_add_3#output")
    # creat To * tang_add_3  #
    out1 = sgs_chalk.Mul(logistic_To, tanh_add_3, name = prefix + "h_n")

    #outfilename = model_config["name"] + "_unroll.sim"
    model = sgs_chalk.Model([in0,in1,in2,in3],out1,model_config["name"])
    buf = model.save_buf()

    #print("\nWell Done! " + outfilename  + " generated!\n")
    return buf


def get_postprocess():

    #load file to parse data
    path = './lstm_data'
    files = os.listdir(path)
    files_num = len(files)
    lstm_num = files_num//2
    outputfile = []
    #1  input&output name and shapes
    for i in six.moves.range(lstm_num):
        file_name = './lstm_data/input_output_shape#' + str(i) + '.npy'
        shape = np.load(file_name,allow_pickle=True)
        prefix = 'sub' + str(lstm_num - 1 - i) + '_'
        for key in shape.item():
            if key.split("_")[-1] == 'input':
              input_name = prefix + 'lstm_input'
              input_shape = shape.item()[key]
            elif key.split("_")[-1] == 'h0':
              h_0 = prefix + 'lstm_h0'
              h0_shape = shape.item()[key]
            elif key.split("_")[-1] == 'c0':
              c_0 = prefix + 'lstm_c0'
              c0_shape = shape.item()[key]
            elif key.split("_")[-1] == 'output':
              output_name = prefix + 'lstm_output'
              output_shape = shape.item()[key]
            elif key.split("_")[-1] == 'time':
              indicator = prefix + 'lstm_time'
              indicator_shape = shape.item()[key]

        modle_name = 'SGS_LSTM_sub' + str(lstm_num - 1 - i)
        model_config = {"name":modle_name,
              "input" : [input_name,indicator,h_0,c_0,],
              "input_shape" : [input_shape,indicator_shape,h0_shape,c0_shape],
              "out_shapes" : [output_shape]}
        sub_buf = buildGraph(model_config,lstm_num - 1 - i,lstm_num)
        outputfile.append(sub_buf)
    return outputfile

def model_postprocess():
    return get_postprocess()
