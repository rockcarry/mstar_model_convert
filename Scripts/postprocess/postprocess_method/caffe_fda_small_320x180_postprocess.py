from TFLitePostProcess import *
from  anchor_param import *
from third_party import tflite
from itertools import product as product
from math import ceil
import numpy as np

def buildGraph(sgs_builder,model_config):
    """

    :return:
    """
    """=========================================="""
    fda_anchors = anchors_generate(model_config["input_hw"][1], model_config["input_hw"][0])
    offset = model_config["shape"][1]
    for j in range(4):
        anchor_vector=[]
        for i in range(offset):
            anchor_vector += bytearray(struct.pack("f", fda_anchors[i][j]))
        buildBufferName = "anchor"+str(j)+"_buffer"
        buildTensorName = "anchor"+str(j)+"_tensor"
        sgs_builder.buildBuffer(buildBufferName, anchor_vector)
        sgs_builder.buildTensor([offset],buildTensorName,sgs_builder.getBufferByName(buildBufferName))

    variances = [0.1, 0.2]
    for i in range(2):
        variances_vector=[]
        variances_vector += bytearray(struct.pack("f", variances[i]))
        buildBufferName = "variances"+str(i)+"_buffer"
        buildTensorName = "variances"+str(i)+"_tensor"
        sgs_builder.buildBuffer(buildBufferName, variances_vector)
        sgs_builder.buildTensor([1],buildTensorName,sgs_builder.getBufferByName(buildBufferName))

    half = [0.5]
    half_vector=[]
    half_vector += bytearray(struct.pack("f", half[0]))
    sgs_builder.buildBuffer("half_buffer", half_vector)
    sgs_builder.buildTensor([1],"half_tensor",sgs_builder.getBufferByName("half_buffer"))
    """=========================================="""
    unpack_in_tensors1 = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][0],model_config["input"][0])
    unpack_in_tensors1.append(model_config["input"][0])
    unpack_out_tensors1 = []
    for i in range(4):
        sgs_builder.buildTensor(model_config["shape"],"SGS_unpack1_"+str(i))
        unpack_out_tensors1.append("SGS_unpack1_"+str(i))
    sgs_builder.buildOperatorCode("SGS_unpack1",tflite.BuiltinOperator.BuiltinOperator().UNPACK)
    unpack_optionts1 = sgs_builder.createUnpackOptions(4, 2)
    sgs_builder.buildOperator("SGS_unpack1",unpack_in_tensors1,unpack_out_tensors1,tflite.BuiltinOptions.BuiltinOptions().UnpackOptions,unpack_optionts1)

    unpack10_multiv0_out_tensors = []
    unpack10_multiv0_in_tensors = []
    unpack10_multiv0_in_tensors.append("SGS_unpack1_0")
    unpack10_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack10_multiv0_tensor")
    unpack10_multiv0_out_tensors.append("unpack10_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack10_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack10_multi0",unpack10_multiv0_in_tensors,unpack10_multiv0_out_tensors)

    unpack10_multiv0_mula2_out_tensors = []
    unpack10_multiv0_mula2_in_tensors = []
    unpack10_multiv0_mula2_in_tensors.append("unpack10_multiv0_tensor")
    unpack10_multiv0_mula2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack10_multiv0_mula2_tensor")
    unpack10_multiv0_mula2_out_tensors.append("unpack10_multiv0_mula2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack10_multi0_mula2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack10_multi0_mula2",unpack10_multiv0_mula2_in_tensors,unpack10_multiv0_mula2_out_tensors)

    unpack10_multiv0_mula2_adda0_out_tensors = []
    unpack10_multiv0_mula2_adda0_in_tensors = []
    unpack10_multiv0_mula2_adda0_in_tensors.append("unpack10_multiv0_mula2_tensor")
    unpack10_multiv0_mula2_adda0_in_tensors.append("anchor0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack10_multiv0_mula2_adda0_tensor")
    unpack10_multiv0_mula2_adda0_out_tensors.append("unpack10_multiv0_mula2_adda0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack10_multiv0_mula2_adda0",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack10_multiv0_mula2_adda0",unpack10_multiv0_mula2_adda0_in_tensors,unpack10_multiv0_mula2_adda0_out_tensors)


    unpack11_multiv0_out_tensors = []
    unpack11_multiv0_in_tensors = []
    unpack11_multiv0_in_tensors.append("SGS_unpack1_1")
    unpack11_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack11_multiv0_tensor")
    unpack11_multiv0_out_tensors.append("unpack11_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack11_multi0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack11_multi0",unpack11_multiv0_in_tensors,unpack11_multiv0_out_tensors)

    unpack11_multiv0_mula3_out_tensors = []
    unpack11_multiv0_mula3_in_tensors = []
    unpack11_multiv0_mula3_in_tensors.append("unpack11_multiv0_tensor")
    unpack11_multiv0_mula3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack11_multiv0_mula3_tensor")
    unpack11_multiv0_mula3_out_tensors.append("unpack11_multiv0_mula3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack11_multi0_mula3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack11_multi0_mula3",unpack11_multiv0_mula3_in_tensors,unpack11_multiv0_mula3_out_tensors)

    unpack11_multiv0_mula3_adda1_out_tensors = []
    unpack11_multiv0_mula3_adda1_in_tensors = []
    unpack11_multiv0_mula3_adda1_in_tensors.append("unpack11_multiv0_mula3_tensor")
    unpack11_multiv0_mula3_adda1_in_tensors.append("anchor1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack11_multiv0_mula3_adda1_tensor")
    unpack11_multiv0_mula3_adda1_out_tensors.append("unpack11_multiv0_mula3_adda1_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack11_multiv0_mula3_adda1",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack11_multiv0_mula3_adda1",unpack11_multiv0_mula3_adda1_in_tensors,unpack11_multiv0_mula3_adda1_out_tensors)


    unpack12_multiv1_out_tensors = []
    unpack12_multiv1_in_tensors = []
    unpack12_multiv1_in_tensors.append("SGS_unpack1_2")
    unpack12_multiv1_in_tensors.append("variances1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack12_multiv1_tensor")
    unpack12_multiv1_out_tensors.append("unpack12_multiv1_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack12_multiv1",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack12_multiv1",unpack12_multiv1_in_tensors,unpack12_multiv1_out_tensors)

    unpack12_multiv1_exp_out_tensors = []
    unpack12_multiv1_exp_in_tensors = []
    unpack12_multiv1_exp_in_tensors.append("unpack12_multiv1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack12_multiv1_exp_tensor")
    unpack12_multiv1_exp_out_tensors.append("unpack12_multiv1_exp_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack12_multiv1_exp",tflite.BuiltinOperator.BuiltinOperator().EXP)
    sgs_builder.buildOperator("SGS_unpack12_multiv1_exp",unpack12_multiv1_exp_in_tensors,unpack12_multiv1_exp_out_tensors)

    unpack12_multiv1_exp_mula2_out_tensors = []
    unpack12_multiv1_exp_mula2_in_tensors = []
    unpack12_multiv1_exp_mula2_in_tensors.append("unpack12_multiv1_exp_tensor")
    unpack12_multiv1_exp_mula2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack12_multiv1_exp_mula2_tensor")
    unpack12_multiv1_exp_mula2_out_tensors.append("unpack12_multiv1_exp_mula2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack12_multiv1_exp_mula2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack12_multiv1_exp_mula2",unpack12_multiv1_exp_mula2_in_tensors,unpack12_multiv1_exp_mula2_out_tensors)

    unpack12_multiv1_exp_mula2_half_out_tensors = []
    unpack12_multiv1_exp_mula2_half_in_tensors = []
    unpack12_multiv1_exp_mula2_half_in_tensors.append("unpack12_multiv1_exp_mula2_tensor")
    unpack12_multiv1_exp_mula2_half_in_tensors.append("half_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack12_multiv1_exp_mula2_half_tensor")
    unpack12_multiv1_exp_mula2_half_out_tensors.append("unpack12_multiv1_exp_mula2_half_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack12_multiv1_exp_mula2_half",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack12_multiv1_exp_mula2_half",unpack12_multiv1_exp_mula2_half_in_tensors,unpack12_multiv1_exp_mula2_half_out_tensors)

    unpack13_multiv1_out_tensors = []
    unpack13_multiv1_in_tensors = []
    unpack13_multiv1_in_tensors.append("SGS_unpack1_3")
    unpack13_multiv1_in_tensors.append("variances1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack13_multiv1_tensor")
    unpack13_multiv1_out_tensors.append("unpack13_multiv1_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack13_multiv1",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack13_multiv1",unpack13_multiv1_in_tensors,unpack13_multiv1_out_tensors)

    unpack13_multiv1_exp_out_tensors = []
    unpack13_multiv1_exp_in_tensors = []
    unpack13_multiv1_exp_in_tensors.append("unpack13_multiv1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack13_multiv1_exp_tensor")
    unpack13_multiv1_exp_out_tensors.append("unpack13_multiv1_exp_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack13_multiv1_exp",tflite.BuiltinOperator.BuiltinOperator().EXP)
    sgs_builder.buildOperator("SGS_unpack13_multiv1_exp",unpack13_multiv1_exp_in_tensors,unpack13_multiv1_exp_out_tensors)

    unpack13_multiv1_exp_mula3_out_tensors = []
    unpack13_multiv1_exp_mula3_in_tensors = []
    unpack13_multiv1_exp_mula3_in_tensors.append("unpack13_multiv1_exp_tensor")
    unpack13_multiv1_exp_mula3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack13_multiv1_exp_mula3_tensor")
    unpack13_multiv1_exp_mula3_out_tensors.append("unpack13_multiv1_exp_mula3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack13_multiv1_exp_mula3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack13_multiv1_exp_mula3",unpack13_multiv1_exp_mula3_in_tensors,unpack13_multiv1_exp_mula3_out_tensors)

    unpack13_multiv1_exp_mula3_half_out_tensors = []
    unpack13_multiv1_exp_mula3_half_in_tensors = []
    unpack13_multiv1_exp_mula3_half_in_tensors.append("unpack13_multiv1_exp_mula3_tensor")
    unpack13_multiv1_exp_mula3_half_in_tensors.append("half_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack13_multiv1_exp_mula3_half_tensor")
    unpack13_multiv1_exp_mula3_half_out_tensors.append("unpack13_multiv1_exp_mula3_half_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack13_multiv1_exp_mula3_half",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack13_multiv1_exp_mula3_half",unpack13_multiv1_exp_mula3_half_in_tensors,unpack13_multiv1_exp_mula3_half_out_tensors)

    x1_out_tensors = []
    x1_in_tensors = []
    x1_in_tensors.append("unpack10_multiv0_mula2_adda0_tensor")
    x1_in_tensors.append("unpack12_multiv1_exp_mula2_half_tensor")
    sgs_builder.buildTensor(model_config["shape"],"x1_tensor")
    x1_out_tensors.append("x1_tensor")
    sgs_builder.buildOperatorCode("SGS_x1_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
    sgs_builder.buildOperator("SGS_x1_sub",x1_in_tensors,x1_out_tensors)

    y1_out_tensors = []
    y1_in_tensors = []
    y1_in_tensors.append("unpack11_multiv0_mula3_adda1_tensor")
    y1_in_tensors.append("unpack13_multiv1_exp_mula3_half_tensor")
    sgs_builder.buildTensor(model_config["shape"],"y1_tensor")
    y1_out_tensors.append("y1_tensor")
    sgs_builder.buildOperatorCode("SGS_y1_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
    sgs_builder.buildOperator("SGS_y1_sub",y1_in_tensors,y1_out_tensors)

    x2_out_tensors = []
    x2_in_tensors = []
    x2_in_tensors.append("unpack10_multiv0_mula2_adda0_tensor")
    x2_in_tensors.append("unpack12_multiv1_exp_mula2_half_tensor")
    sgs_builder.buildTensor(model_config["shape"],"x2_tensor")
    x2_out_tensors.append("x2_tensor")
    sgs_builder.buildOperatorCode("SGS_x2_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_x2_add",x2_in_tensors,x2_out_tensors)

    y2_out_tensors = []
    y2_in_tensors = []
    y2_in_tensors.append("unpack11_multiv0_mula3_adda1_tensor")
    y2_in_tensors.append("unpack13_multiv1_exp_mula3_half_tensor")
    sgs_builder.buildTensor(model_config["shape"],"y2_tensor")
    y2_out_tensors.append("y2_tensor")
    sgs_builder.buildOperatorCode("SGS_y2_add",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_y2_add",y2_in_tensors,y2_out_tensors)


    """=========================================="""
    unpack_in_tensors2 = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][1],model_config["input"][1])
    unpack_in_tensors2.append(model_config["input"][1])
    unpack_out_tensors2 = []
    for i in range(2):
        sgs_builder.buildTensor(model_config["shape"],"unpack2_"+str(i))
        unpack_out_tensors2.append("unpack2_"+str(i))
    sgs_builder.buildOperatorCode("SGS_unpack2",tflite.BuiltinOperator.BuiltinOperator().UNPACK)
    unpack_optionts2 = sgs_builder.createUnpackOptions(2, 2)
    sgs_builder.buildOperator("SGS_unpack2",unpack_in_tensors2,unpack_out_tensors2,tflite.BuiltinOptions.BuiltinOptions().UnpackOptions,unpack_optionts2)

    unpack2_sub_out_tensors = []
    unpack2_sub_in_tensors = []
    unpack2_sub_in_tensors.append("unpack2_1")
    unpack2_sub_in_tensors.append("unpack2_0")
    sgs_builder.buildTensor(model_config["shape"],"unpack2_sub_tensor")
    unpack2_sub_out_tensors.append("unpack2_sub_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack2_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
    sgs_builder.buildOperator("SGS_unpack2_sub",unpack2_sub_in_tensors,unpack2_sub_out_tensors)

    unpack2_sub_logistic_out_tensors = []
    unpack2_sub_logistic_in_tensors = []
    unpack2_sub_logistic_in_tensors.append("unpack2_sub_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack2_sub_logistic_tensor")
    unpack2_sub_logistic_out_tensors.append("unpack2_sub_logistic_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack2_sub_logistic",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("SGS_unpack2_sub_logistic",unpack2_sub_logistic_in_tensors,unpack2_sub_logistic_out_tensors)

    """=========================================="""
    transpose_in_tensors = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][2],model_config["input"][2])
    transpose_in_tensors.append(model_config["input"][2])
    transpose_vector_val = [0,2,1]
    transpose_vector=[]
    for value in transpose_vector_val:
        transpose_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("transpose_vector",transpose_vector)
    sgs_builder.buildTensor([len(transpose_vector_val)],"transpose_shape",sgs_builder.getBufferByName("transpose_vector"),tflite.TensorType.TensorType().INT32)
    transpose_in_tensors.append("transpose_shape")
    transpose_out_shape =  [1,10,3405]
    transpose_out_tensors = []
    sgs_builder.buildTensor(transpose_out_shape,"transpose_tensor")
    transpose_out_tensors.append("transpose_tensor")
    sgs_builder.buildOperatorCode("SGS_transpose",tflite.BuiltinOperator.BuiltinOperator().TRANSPOSE)
    sgs_builder.buildOperator("SGS_transpose",transpose_in_tensors, transpose_out_tensors)

    unpack_out_tensors3 = []
    for i in range(10):
        sgs_builder.buildTensor(model_config["shape"],"SGS_unpack3_"+str(i))
        unpack_out_tensors3.append("SGS_unpack3_"+str(i))
    sgs_builder.buildOperatorCode("SGS_unpack3",tflite.BuiltinOperator.BuiltinOperator().UNPACK)
    unpack_optionts3 = sgs_builder.createUnpackOptions(10, 1)
    sgs_builder.buildOperator("SGS_unpack3",transpose_out_tensors,unpack_out_tensors3,tflite.BuiltinOptions.BuiltinOptions().UnpackOptions,unpack_optionts3)

    unpack30_multiv0_out_tensors = []
    unpack30_multiv0_in_tensors = []
    unpack30_multiv0_in_tensors.append("SGS_unpack3_0")
    unpack30_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack30_multiv0_tensor")
    unpack30_multiv0_out_tensors.append("unpack30_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack30_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack30_multiv0",unpack30_multiv0_in_tensors,unpack30_multiv0_out_tensors)

    unpack31_multiv0_out_tensors = []
    unpack31_multiv0_in_tensors = []
    unpack31_multiv0_in_tensors.append("SGS_unpack3_1")
    unpack31_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack31_multiv0_tensor")
    unpack31_multiv0_out_tensors.append("unpack31_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack31_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack31_multiv0",unpack31_multiv0_in_tensors,unpack31_multiv0_out_tensors)

    unpack32_multiv0_out_tensors = []
    unpack32_multiv0_in_tensors = []
    unpack32_multiv0_in_tensors.append("SGS_unpack3_2")
    unpack32_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack32_multiv0_tensor")
    unpack32_multiv0_out_tensors.append("unpack32_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack32_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack32_multiv0",unpack32_multiv0_in_tensors,unpack32_multiv0_out_tensors)

    unpack33_multiv0_out_tensors = []
    unpack33_multiv0_in_tensors = []
    unpack33_multiv0_in_tensors.append("SGS_unpack3_3")
    unpack33_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack33_multiv0_tensor")
    unpack33_multiv0_out_tensors.append("unpack33_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack33_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack33_multiv0",unpack33_multiv0_in_tensors,unpack33_multiv0_out_tensors)

    unpack34_multiv0_out_tensors = []
    unpack34_multiv0_in_tensors = []
    unpack34_multiv0_in_tensors.append("SGS_unpack3_4")
    unpack34_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack34_multiv0_tensor")
    unpack34_multiv0_out_tensors.append("unpack34_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack34_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack34_multiv0",unpack34_multiv0_in_tensors,unpack34_multiv0_out_tensors)

    unpack35_multiv0_out_tensors = []
    unpack35_multiv0_in_tensors = []
    unpack35_multiv0_in_tensors.append("SGS_unpack3_5")
    unpack35_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack35_multiv0_tensor")
    unpack35_multiv0_out_tensors.append("unpack35_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack35_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack35_multiv0",unpack35_multiv0_in_tensors,unpack35_multiv0_out_tensors)

    unpack36_multiv0_out_tensors = []
    unpack36_multiv0_in_tensors = []
    unpack36_multiv0_in_tensors.append("SGS_unpack3_6")
    unpack36_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack36_multiv0_tensor")
    unpack36_multiv0_out_tensors.append("unpack36_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack36_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack36_multiv0",unpack36_multiv0_in_tensors,unpack36_multiv0_out_tensors)

    unpack37_multiv0_out_tensors = []
    unpack37_multiv0_in_tensors = []
    unpack37_multiv0_in_tensors.append("SGS_unpack3_7")
    unpack37_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack37_multiv0_tensor")
    unpack37_multiv0_out_tensors.append("unpack37_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack37_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack37_multiv0",unpack37_multiv0_in_tensors,unpack37_multiv0_out_tensors)

    unpack38_multiv0_out_tensors = []
    unpack38_multiv0_in_tensors = []
    unpack38_multiv0_in_tensors.append("SGS_unpack3_8")
    unpack38_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack38_multiv0_tensor")
    unpack38_multiv0_out_tensors.append("unpack38_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack38_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack38_multiv0",unpack38_multiv0_in_tensors,unpack38_multiv0_out_tensors)

    unpack39_multiv0_out_tensors = []
    unpack39_multiv0_in_tensors = []
    unpack39_multiv0_in_tensors.append("SGS_unpack3_9")
    unpack39_multiv0_in_tensors.append("variances0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack39_multiv0_tensor")
    unpack39_multiv0_out_tensors.append("unpack39_multiv0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack39_multiv0",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack39_multiv0",unpack39_multiv0_in_tensors,unpack39_multiv0_out_tensors)



    unpack30_multiv0_mulanchor2_out_tensors = []
    unpack30_multiv0_mulanchor2_in_tensors = []
    unpack30_multiv0_mulanchor2_in_tensors.append("unpack30_multiv0_tensor")
    unpack30_multiv0_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack30_multiv0_mulanchor2_tensor")
    unpack30_multiv0_mulanchor2_out_tensors.append("unpack30_multiv0_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack30_multiv0_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack30_multiv0_mulanchor2",unpack30_multiv0_mulanchor2_in_tensors,unpack30_multiv0_mulanchor2_out_tensors)

    unpack31_multiv0_mulanchor3_out_tensors = []
    unpack31_multiv0_mulanchor3_in_tensors = []
    unpack31_multiv0_mulanchor3_in_tensors.append("unpack31_multiv0_tensor")
    unpack31_multiv0_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack31_multiv0_mulanchor3_tensor")
    unpack31_multiv0_mulanchor3_out_tensors.append("unpack31_multiv0_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack31_multiv0_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack31_multiv0_mulanchor3",unpack31_multiv0_mulanchor3_in_tensors,unpack31_multiv0_mulanchor3_out_tensors)

    unpack32_multiv0_mulanchor2_out_tensors = []
    unpack32_multiv0_mulanchor2_in_tensors = []
    unpack32_multiv0_mulanchor2_in_tensors.append("unpack32_multiv0_tensor")
    unpack32_multiv0_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack32_multiv0_mulanchor2_tensor")
    unpack32_multiv0_mulanchor2_out_tensors.append("unpack32_multiv0_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack32_multiv0_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack32_multiv0_mulanchor2",unpack32_multiv0_mulanchor2_in_tensors,unpack32_multiv0_mulanchor2_out_tensors)

    unpack33_multiv0_mulanchor3_out_tensors = []
    unpack33_multiv0_mulanchor3_in_tensors = []
    unpack33_multiv0_mulanchor3_in_tensors.append("unpack33_multiv0_tensor")
    unpack33_multiv0_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack33_multiv0_mulanchor3_tensor")
    unpack33_multiv0_mulanchor3_out_tensors.append("unpack33_multiv0_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack33_multiv0_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack33_multiv0_mulanchor3",unpack33_multiv0_mulanchor3_in_tensors,unpack33_multiv0_mulanchor3_out_tensors)

    unpack34_multiv0_mulanchor2_out_tensors = []
    unpack34_multiv0_mulanchor2_in_tensors = []
    unpack34_multiv0_mulanchor2_in_tensors.append("unpack34_multiv0_tensor")
    unpack34_multiv0_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack34_multiv0_mulanchor2_tensor")
    unpack34_multiv0_mulanchor2_out_tensors.append("unpack34_multiv0_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack34_multiv0_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack34_multiv0_mulanchor2",unpack34_multiv0_mulanchor2_in_tensors,unpack34_multiv0_mulanchor2_out_tensors)

    unpack35_multiv0_mulanchor3_out_tensors = []
    unpack35_multiv0_mulanchor3_in_tensors = []
    unpack35_multiv0_mulanchor3_in_tensors.append("unpack35_multiv0_tensor")
    unpack35_multiv0_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack35_multiv0_mulanchor3_tensor")
    unpack35_multiv0_mulanchor3_out_tensors.append("unpack35_multiv0_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack35_multiv0_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack35_multiv0_mulanchor3",unpack35_multiv0_mulanchor3_in_tensors,unpack35_multiv0_mulanchor3_out_tensors)

    unpack36_multiv0_mulanchor2_out_tensors = []
    unpack36_multiv0_mulanchor2_in_tensors = []
    unpack36_multiv0_mulanchor2_in_tensors.append("unpack36_multiv0_tensor")
    unpack36_multiv0_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack36_multiv0_mulanchor2_tensor")
    unpack36_multiv0_mulanchor2_out_tensors.append("unpack36_multiv0_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack36_multiv0_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack36_multiv0_mulanchor2",unpack36_multiv0_mulanchor2_in_tensors,unpack36_multiv0_mulanchor2_out_tensors)

    unpack37_multiv0_mulanchor3_out_tensors = []
    unpack37_multiv0_mulanchor3_in_tensors = []
    unpack37_multiv0_mulanchor3_in_tensors.append("unpack37_multiv0_tensor")
    unpack37_multiv0_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack37_multiv0_mulanchor3_tensor")
    unpack37_multiv0_mulanchor3_out_tensors.append("unpack37_multiv0_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack37_multiv0_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack37_multiv0_mulanchor3",unpack37_multiv0_mulanchor3_in_tensors,unpack37_multiv0_mulanchor3_out_tensors)

    unpack38_multiv0_mulanchor2_out_tensors = []
    unpack38_multiv0_mulanchor2_in_tensors = []
    unpack38_multiv0_mulanchor2_in_tensors.append("unpack38_multiv0_tensor")
    unpack38_multiv0_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack38_multiv0_mulanchor2_tensor")
    unpack38_multiv0_mulanchor2_out_tensors.append("unpack38_multiv0_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack38_multiv0_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack38_multiv0_mulanchor2",unpack38_multiv0_mulanchor2_in_tensors,unpack38_multiv0_mulanchor2_out_tensors)

    unpack39_multiv0_mulanchor3_out_tensors = []
    unpack39_multiv0_mulanchor3_in_tensors = []
    unpack39_multiv0_mulanchor3_in_tensors.append("unpack39_multiv0_tensor")
    unpack39_multiv0_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack39_multiv0_mulanchor3_tensor")
    unpack39_multiv0_mulanchor3_out_tensors.append("unpack39_multiv0_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack39_multiv0_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack39_multiv0_mulanchor3",unpack39_multiv0_mulanchor3_in_tensors,unpack39_multiv0_mulanchor3_out_tensors)



    unpack30_multiv0_mulanchor2_addanchor0_out_tensors = []
    unpack30_multiv0_mulanchor2_addanchor0_in_tensors = []
    unpack30_multiv0_mulanchor2_addanchor0_in_tensors.append("unpack30_multiv0_mulanchor2_tensor")
    unpack30_multiv0_mulanchor2_addanchor0_in_tensors.append("anchor0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack30_multiv0_mulanchor2_addanchor0_tensor")
    unpack30_multiv0_mulanchor2_addanchor0_out_tensors.append("unpack30_multiv0_mulanchor2_addanchor0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack30_multiv0_mulanchor2_addanchor0",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack30_multiv0_mulanchor2_addanchor0",unpack30_multiv0_mulanchor2_addanchor0_in_tensors,unpack30_multiv0_mulanchor2_addanchor0_out_tensors)

    unpack31_multiv0_mulanchor3_addanchor1_out_tensors = []
    unpack31_multiv0_mulanchor3_addanchor1_in_tensors = []
    unpack31_multiv0_mulanchor3_addanchor1_in_tensors.append("unpack31_multiv0_mulanchor3_tensor")
    unpack31_multiv0_mulanchor3_addanchor1_in_tensors.append("anchor1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack31_multiv0_mulanchor3_addanchor1_tensor")
    unpack31_multiv0_mulanchor3_addanchor1_out_tensors.append("unpack31_multiv0_mulanchor3_addanchor1_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack31_multiv0_mulanchor3_addanchor1",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack31_multiv0_mulanchor3_addanchor1",unpack31_multiv0_mulanchor3_addanchor1_in_tensors,unpack31_multiv0_mulanchor3_addanchor1_out_tensors)

    unpack32_multiv0_mulanchor2_addanchor0_out_tensors = []
    unpack32_multiv0_mulanchor2_addanchor0_in_tensors = []
    unpack32_multiv0_mulanchor2_addanchor0_in_tensors.append("unpack32_multiv0_mulanchor2_tensor")
    unpack32_multiv0_mulanchor2_addanchor0_in_tensors.append("anchor0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack32_multiv0_mulanchor2_addanchor0_tensor")
    unpack32_multiv0_mulanchor2_addanchor0_out_tensors.append("unpack32_multiv0_mulanchor2_addanchor0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack32_multiv0_mulanchor2_addanchor0",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack32_multiv0_mulanchor2_addanchor0",unpack32_multiv0_mulanchor2_addanchor0_in_tensors,unpack32_multiv0_mulanchor2_addanchor0_out_tensors)

    unpack33_multiv0_mulanchor3_addanchor1_out_tensors = []
    unpack33_multiv0_mulanchor3_addanchor1_in_tensors = []
    unpack33_multiv0_mulanchor3_addanchor1_in_tensors.append("unpack33_multiv0_mulanchor3_tensor")
    unpack33_multiv0_mulanchor3_addanchor1_in_tensors.append("anchor1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack33_multiv0_mulanchor3_addanchor1_tensor")
    unpack33_multiv0_mulanchor3_addanchor1_out_tensors.append("unpack33_multiv0_mulanchor3_addanchor1_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack33_multiv0_mulanchor3_addanchor1",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack33_multiv0_mulanchor3_addanchor1",unpack33_multiv0_mulanchor3_addanchor1_in_tensors,unpack33_multiv0_mulanchor3_addanchor1_out_tensors)

    unpack34_multiv0_mulanchor2_addanchor0_out_tensors = []
    unpack34_multiv0_mulanchor2_addanchor0_in_tensors = []
    unpack34_multiv0_mulanchor2_addanchor0_in_tensors.append("unpack34_multiv0_mulanchor2_tensor")
    unpack34_multiv0_mulanchor2_addanchor0_in_tensors.append("anchor0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack34_multiv0_mulanchor2_addanchor0_tensor")
    unpack34_multiv0_mulanchor2_addanchor0_out_tensors.append("unpack34_multiv0_mulanchor2_addanchor0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack34_multiv0_mulanchor2_addanchor0",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack34_multiv0_mulanchor2_addanchor0",unpack34_multiv0_mulanchor2_addanchor0_in_tensors,unpack34_multiv0_mulanchor2_addanchor0_out_tensors)

    unpack35_multiv0_mulanchor3_addanchor1_out_tensors = []
    unpack35_multiv0_mulanchor3_addanchor1_in_tensors = []
    unpack35_multiv0_mulanchor3_addanchor1_in_tensors.append("unpack35_multiv0_mulanchor3_tensor")
    unpack35_multiv0_mulanchor3_addanchor1_in_tensors.append("anchor1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack35_multiv0_mulanchor3_addanchor1_tensor")
    unpack35_multiv0_mulanchor3_addanchor1_out_tensors.append("unpack35_multiv0_mulanchor3_addanchor1_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack35_multiv0_mulanchor3_addanchor1",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack35_multiv0_mulanchor3_addanchor1",unpack35_multiv0_mulanchor3_addanchor1_in_tensors,unpack35_multiv0_mulanchor3_addanchor1_out_tensors)

    unpack36_multiv0_mulanchor2_addanchor0_out_tensors = []
    unpack36_multiv0_mulanchor2_addanchor0_in_tensors = []
    unpack36_multiv0_mulanchor2_addanchor0_in_tensors.append("unpack36_multiv0_mulanchor2_tensor")
    unpack36_multiv0_mulanchor2_addanchor0_in_tensors.append("anchor0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack36_multiv0_mulanchor2_addanchor0_tensor")
    unpack36_multiv0_mulanchor2_addanchor0_out_tensors.append("unpack36_multiv0_mulanchor2_addanchor0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack36_multiv0_mulanchor2_addanchor0",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack36_multiv0_mulanchor2_addanchor0",unpack36_multiv0_mulanchor2_addanchor0_in_tensors,unpack36_multiv0_mulanchor2_addanchor0_out_tensors)

    unpack37_multiv0_mulanchor3_addanchor1_out_tensors = []
    unpack37_multiv0_mulanchor3_addanchor1_in_tensors = []
    unpack37_multiv0_mulanchor3_addanchor1_in_tensors.append("unpack37_multiv0_mulanchor3_tensor")
    unpack37_multiv0_mulanchor3_addanchor1_in_tensors.append("anchor1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack37_multiv0_mulanchor3_addanchor1_tensor")
    unpack37_multiv0_mulanchor3_addanchor1_out_tensors.append("unpack37_multiv0_mulanchor3_addanchor1_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack37_multiv0_mulanchor3_addanchor1",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack37_multiv0_mulanchor3_addanchor1",unpack37_multiv0_mulanchor3_addanchor1_in_tensors,unpack37_multiv0_mulanchor3_addanchor1_out_tensors)

    unpack38_multiv0_mulanchor2_addanchor0_out_tensors = []
    unpack38_multiv0_mulanchor2_addanchor0_in_tensors = []
    unpack38_multiv0_mulanchor2_addanchor0_in_tensors.append("unpack38_multiv0_mulanchor2_tensor")
    unpack38_multiv0_mulanchor2_addanchor0_in_tensors.append("anchor0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack38_multiv0_mulanchor2_addanchor0_tensor")
    unpack38_multiv0_mulanchor2_addanchor0_out_tensors.append("unpack38_multiv0_mulanchor2_addanchor0_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack38_multiv0_mulanchor2_addanchor0",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack38_multiv0_mulanchor2_addanchor0",unpack38_multiv0_mulanchor2_addanchor0_in_tensors,unpack38_multiv0_mulanchor2_addanchor0_out_tensors)

    unpack39_multiv0_mulanchor3_addanchor1_out_tensors = []
    unpack39_multiv0_mulanchor3_addanchor1_in_tensors = []
    unpack39_multiv0_mulanchor3_addanchor1_in_tensors.append("unpack39_multiv0_mulanchor3_tensor")
    unpack39_multiv0_mulanchor3_addanchor1_in_tensors.append("anchor1_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack39_multiv0_mulanchor3_addanchor1_tensor")
    unpack39_multiv0_mulanchor3_addanchor1_out_tensors.append("unpack39_multiv0_mulanchor3_addanchor1_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack39_multiv0_mulanchor3_addanchor1",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack39_multiv0_mulanchor3_addanchor1",unpack39_multiv0_mulanchor3_addanchor1_in_tensors,unpack39_multiv0_mulanchor3_addanchor1_out_tensors)


    concat_in_tensors = []
    concat_in_tensors.append("unpack30_multiv0_mulanchor2_addanchor0_tensor")
    concat_in_tensors.append("unpack31_multiv0_mulanchor3_addanchor1_tensor")
    concat_in_tensors.append("unpack32_multiv0_mulanchor2_addanchor0_tensor")
    concat_in_tensors.append("unpack33_multiv0_mulanchor3_addanchor1_tensor")
    concat_in_tensors.append("unpack34_multiv0_mulanchor2_addanchor0_tensor")
    concat_in_tensors.append("unpack35_multiv0_mulanchor3_addanchor1_tensor")
    concat_in_tensors.append("unpack36_multiv0_mulanchor2_addanchor0_tensor")
    concat_in_tensors.append("unpack37_multiv0_mulanchor3_addanchor1_tensor")
    concat_in_tensors.append("unpack38_multiv0_mulanchor2_addanchor0_tensor")
    concat_in_tensors.append("unpack39_multiv0_mulanchor3_addanchor1_tensor")

    concat_out_tensors = []
    sgs_builder.buildTensor([10,3405],"lms_concat")
    concat_out_tensors.append("lms_concat")
    sgs_builder.buildOperatorCode("SGS_lms_concat",tflite.BuiltinOperator.BuiltinOperator().CONCATENATION)
    concat_optionts = sgs_builder.createConcatenationOptions(0, 0)
    sgs_builder.buildOperator("SGS_lms_concat",concat_in_tensors,concat_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions,concat_optionts)

    concat_transpose_in_tensors=[]
    concat_transpose_in_tensors.append("lms_concat")
    concat_transpose_vector_val = [1,0]
    concat_transpose_vector=[]
    for value in concat_transpose_vector_val:
        concat_transpose_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("concat_transpose_vector",concat_transpose_vector)
    sgs_builder.buildTensor([len(concat_transpose_vector_val)],"concat_transpose_shape",sgs_builder.getBufferByName("concat_transpose_vector"),tflite.TensorType.TensorType().INT32)
    concat_transpose_in_tensors.append("concat_transpose_shape")
    concat_transpose_out_shape =  [3405,10]
    concat_transpose_out_tensors = []
    sgs_builder.buildTensor(concat_transpose_out_shape,"concat_transpose_tensor")
    concat_transpose_out_tensors.append("concat_transpose_tensor")
    sgs_builder.buildOperatorCode("SGS_concat_transpose",tflite.BuiltinOperator.BuiltinOperator().TRANSPOSE)
    sgs_builder.buildOperator("SGS_concat_transpose",concat_transpose_in_tensors, concat_transpose_out_tensors)


    """=========================================="""
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][1],model_config["input"][1])

    nms_out_tensors = []
    nms_in_tensors = []
    nms_in_tensors.append("x1_tensor")
    nms_in_tensors.append("x2_tensor")
    nms_in_tensors.append("y1_tensor")
    nms_in_tensors.append("y2_tensor")
    nms_in_tensors.append("unpack2_sub_logistic_tensor")

    sgs_builder.buildTensor(model_config["out_shapes"][0],"detectionBoxes")
    nms_out_tensors.append("detectionBoxes")
    sgs_builder.buildTensor(model_config["out_shapes"][1],"detectionClasses")
    nms_out_tensors.append("detectionClasses")
    sgs_builder.buildTensor(model_config["out_shapes"][2],"detectionScores")
    nms_out_tensors.append("detectionScores")
    sgs_builder.buildTensor(model_config["out_shapes"][3],"numDetections")
    nms_out_tensors.append("numDetections")
    sgs_builder.buildTensor(model_config["out_shapes"][4],"detectionIndex")
    nms_out_tensors.append("detectionIndex")
    cus_code = 'TFLite_Detection_NMS'
    sgs_builder.buildOperatorCode("SGS_nms",tflite.BuiltinOperator.BuiltinOperator().CUSTOM,cus_code)
    '''
    1.????????????tensor ??? coordinate score confidence class facecoordinate ????????????
    2.nms ???????????? fast nms / yolo_nms
    3.????????????????????????
    4.????????????tensor ???  bboxcount bboxcoordinate score class facecoordinate ????????????
    5.???????????????input coordinate clip ???[0,1]
                       (b"input_class_idx",4,"int"),
                   (b"input_score_idx",5,"int"),
                   (b"input_confidence_idx",6,"int"),
    '''

    cus_options = [(b"input_coordinate_x1",0,"int"),
                   (b"input_coordinate_y1",2,"int"),
                   (b"input_coordinate_x2",1,"int"),
                   (b"input_coordinate_y2",3,"int"),
                   (b"input_class_idx",-1,"int"),
                   (b"input_score_idx",4,"int"),
                   (b"input_confidence_idx",-1,"int"),
                   (b"input_facecoordinate_idx",5,"int"),
                   (b"output_detection_boxes_idx",0,"int"),
                   (b"output_detection_classes_idx",1,"int"),
                   (b"output_detection_scores_idx",2,"int"),
                   (b"output_num_detection_idx",3,"int"),
                   (b"output_detection_boxes_index_idx",4,"int"),
                   (b"nms",0,"float"),
                   (b"clip",0,"float"),
                   (b"max_detections",100,"int"),
                   (b"max_classes_per_detection",1,"int"),
                   (b"detections_per_class",1,"int"),
                   (b"num_classes",1,"int"),
                   (b"bmax_score",0,"int"),
                   (b"offline",0,"int"),
                   (b"num_classes_with_background",1,"int"),
                   (b"nms_score_threshold",0.200000003,"float"),
                   (b"nms_iou_threshold",0.300000012,"float")]
    options = sgs_builder.createFlexBuffer( sgs_builder.lib, cus_options)
    sgs_builder.buildOperator("SGS_nms",nms_in_tensors,nms_out_tensors,None,None,options)

    network_out_tensors = []
    network_out_tensors.append("detectionBoxes")
    network_out_tensors.append("detectionClasses")
    network_out_tensors.append("detectionScores")
    network_out_tensors.append("numDetections")
    network_out_tensors.append("detectionIndex")
    network_out_tensors.append("concat_transpose_tensor")

    sgs_builder.subgraphs.append( sgs_builder.buildSubGraph(model_config["input"],network_out_tensors,model_config["name"]))
    sgs_builder.model = sgs_builder.createModel(3,sgs_builder.operator_codes,sgs_builder.subgraphs,model_config["name"],sgs_builder.buffers)
    file_identifier = b'TFL3'
    sgs_builder.builder.Finish(sgs_builder.model, file_identifier)
    buf = sgs_builder.builder.Output()
    return buf

def anchors_generate(h, w):
    _min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    _steps = [8, 16, 32, 64]
    _variance = [0.1, 0.2]
    _clip = False
    _feature_maps = [[ceil(w / step), ceil(h / step)] for step in _steps]
    anchors = []
    for k, f in enumerate(_feature_maps):
        min_sizes = _min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / h
                s_ky = min_size / w
                dense_cx = [x * _steps[k] / h for x in [j + 0.5]]
                dense_cy = [y * _steps[k] / w for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = np.reshape(anchors,(-1,4),order='A')
    print(output.shape)
    if _clip:
        output.clamp_(max=1, min=0)
    return output

def get_postprocess():
    model_config = {"name":"caffe_fda_small_320x180",
          "input" : ['305','314','323'],
          "input_shape" : [[1,3405,4],[1,3405,2],[1,3405,10]],
          "shape" : [1,3405],
          "out_shapes" : [[1,100,4],[1,100],[1,100],[1],[1,100],[1,3405,10]],
          "input_hw": [180., 320.]}

    fda = TFLitePostProcess()
    fda_buf = buildGraph(fda,model_config)
    outfilename = model_config["name"] + "_postprocess.sim"
    with open(outfilename, 'wb') as f:
        f.write(fda_buf)
        f.close()
    print("\nWell Done!" + outfilename  + " generated!\n")

def model_postprocess():
    return get_postprocess()
