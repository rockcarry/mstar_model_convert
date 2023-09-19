from TFLitePostProcess import *
from  anchor_param import *
from third_party import tflite
import pdb
from mace.python.tools.convert_util import getIPUVersion

MAX_DETECTIONS = 100
if getIPUVersion() in ['I6DC']:
    MAX_DETECTIONS = 64

def buildGraph(sgs_builder,model_config):
    """

    :return:
    """
    reshape_out_shape1 =  [1,4,10140]
    reshape_out_tensors1 = []
    reshape_in_tensors1 = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][0],model_config["input"][0])
    reshape_in_tensors1.append(model_config["input"][0])
    reshape_vector1=[]
    for value in reshape_out_shape1:
        reshape_vector1 += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape_vector1",reshape_vector1)
    sgs_builder.buildTensor([len(reshape_out_shape1)],"reshape_shape1",sgs_builder.getBufferByName("reshape_vector1"),tflite.TensorType.TensorType().INT32)
    reshape_in_tensors1.append("reshape_shape1")
    sgs_builder.buildTensor(reshape_out_shape1,"reshape_tensor1")
    reshape_out_tensors1.append("reshape_tensor1")
    sgs_builder.buildOperatorCode("SGS_reshape1",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape_newshape1 = sgs_builder.createReshapeOptions(reshape_out_shape1)
    sgs_builder.buildOperator("SGS_reshape1",reshape_in_tensors1, reshape_out_tensors1,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape_newshape1)

    reshape_out_shape2 =  [1,2,10140]
    reshape_out_tensors2 = []
    reshape_in_tensors2 = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][1],model_config["input"][1])
    reshape_in_tensors2.append(model_config["input"][1])
    reshape_vector2=[]
    for value in reshape_out_shape2:
        reshape_vector2 += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape_vector2",reshape_vector2)
    sgs_builder.buildTensor([len(reshape_out_shape2)],"reshape_shape2",sgs_builder.getBufferByName("reshape_vector2"),tflite.TensorType.TensorType().INT32)
    reshape_in_tensors2.append("reshape_shape2")
    sgs_builder.buildTensor(reshape_out_shape2,"reshape_tensor2")
    reshape_out_tensors2.append("reshape_tensor2")
    sgs_builder.buildOperatorCode("SGS_reshape2",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape_newshape2 = sgs_builder.createReshapeOptions(reshape_out_shape2)
    sgs_builder.buildOperator("SGS_reshape2",reshape_in_tensors2, reshape_out_tensors2,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape_newshape2)

    reshape_out_shape3 =  [1,10,10140]
    reshape_out_tensors3 = []
    reshape_in_tensors3 = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][2],model_config["input"][2])
    reshape_in_tensors3.append(model_config["input"][2])
    reshape_vector3=[]
    for value in reshape_out_shape3:
        reshape_vector3 += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape_vector3",reshape_vector3)
    sgs_builder.buildTensor([len(reshape_out_shape3)],"reshape_shape3",sgs_builder.getBufferByName("reshape_vector3"),tflite.TensorType.TensorType().INT32)
    reshape_in_tensors3.append("reshape_shape3")
    sgs_builder.buildTensor(reshape_out_shape3,"reshape_tensor3")
    reshape_out_tensors3.append("reshape_tensor3")
    sgs_builder.buildOperatorCode("SGS_reshape3",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape_newshape3 = sgs_builder.createReshapeOptions(reshape_out_shape3)
    sgs_builder.buildOperator("SGS_reshape3",reshape_in_tensors3, reshape_out_tensors3,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape_newshape3)

    """=========================================="""
    fda_anchors14 = []
    anchors_generate(model_config["input_hw"][0], model_config["input_hw"][1], fda_anchors14)

    offset = model_config["shape"][1]
    for j in range(14):
        anchor_vector=[]
        for i in range(offset):
            anchor_vector += bytearray(struct.pack("f", fda_anchors14[j+i*14]))
        buildBufferName = "anchor"+str(j)+"_buffer"
        buildTensorName = "anchor"+str(j)+"_tensor"
        sgs_builder.buildBuffer(buildBufferName, anchor_vector)
        sgs_builder.buildTensor([offset],buildTensorName,sgs_builder.getBufferByName(buildBufferName))

    variances = [0.1, 0.2, 0.2]
    for i in range(3):
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
    unpack_out_tensors1 = []
    for i in range(4):
        sgs_builder.buildTensor(model_config["shape"],"SGS_unpack1_"+str(i))
        unpack_out_tensors1.append("SGS_unpack1_"+str(i))
    sgs_builder.buildOperatorCode("SGS_unpack1",tflite.BuiltinOperator.BuiltinOperator().UNPACK)
    _ChangedOutputIndex = anchor.zeros(16)
    _ChangedOutputShape = anchor.zeros(160)

    unpack_optionts1 = sgs_builder.createUnpackOptions(4, 1)
    sgs_builder.buildOperator("SGS_unpack1",reshape_out_tensors1,unpack_out_tensors1,tflite.BuiltinOptions.BuiltinOptions().UnpackOptions,unpack_optionts1)

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

    unpack_out_tensors2 = []
    for i in range(2):
        sgs_builder.buildTensor(model_config["shape"],"SGS_unpack2_"+str(i))
        unpack_out_tensors2.append("SGS_unpack2_"+str(i))
    sgs_builder.buildOperatorCode("SGS_unpack2",tflite.BuiltinOperator.BuiltinOperator().UNPACK)
    unpack_optionts2 = sgs_builder.createUnpackOptions(2, 1)
    sgs_builder.buildOperator("SGS_unpack2",reshape_out_tensors2,unpack_out_tensors2,tflite.BuiltinOptions.BuiltinOptions().UnpackOptions,unpack_optionts2)

    unpack2_sub_out_tensors = []
    unpack2_sub_in_tensors = []
    unpack2_sub_in_tensors.append("SGS_unpack2_1")
    unpack2_sub_in_tensors.append("SGS_unpack2_0")
    sgs_builder.buildTensor(model_config["shape"],"unpack2_sub_tensor")
    unpack2_sub_out_tensors.append("unpack2_sub_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack2_sub_sub",tflite.BuiltinOperator.BuiltinOperator().SUB)
    sgs_builder.buildOperator("SGS_unpack2_sub_sub",unpack2_sub_in_tensors,unpack2_sub_out_tensors)

    unpack2sub_logistic_out_tensors = []
    unpack2sub_logistic_in_tensors = []
    unpack2sub_logistic_in_tensors.append("unpack2_sub_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack2sub_logistic_tensor")
    unpack2sub_logistic_out_tensors.append("unpack2sub_logistic_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack2sub_logistic",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("SGS_unpack2sub_logistic",unpack2sub_logistic_in_tensors,unpack2sub_logistic_out_tensors)



    """=========================================="""

    unpack_out_tensors3 = []
    for i in range(10):
        sgs_builder.buildTensor(model_config["shape"],"SGS_unpack3_"+str(i))
        unpack_out_tensors3.append("SGS_unpack3_"+str(i))
    sgs_builder.buildOperatorCode("SGS_unpack3",tflite.BuiltinOperator.BuiltinOperator().UNPACK)
    unpack_optionts3 = sgs_builder.createUnpackOptions(10, 1)
    sgs_builder.buildOperator("SGS_unpack3",reshape_out_tensors3,unpack_out_tensors3,tflite.BuiltinOptions.BuiltinOptions().UnpackOptions,unpack_optionts3)

    unpack30_multiv2_out_tensors = []
    unpack30_multiv2_in_tensors = []
    unpack30_multiv2_in_tensors.append("SGS_unpack3_0")
    unpack30_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack30_multiv2_tensor")
    unpack30_multiv2_out_tensors.append("unpack30_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack30_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack30_multiv2",unpack30_multiv2_in_tensors,unpack30_multiv2_out_tensors)

    unpack31_multiv2_out_tensors = []
    unpack31_multiv2_in_tensors = []
    unpack31_multiv2_in_tensors.append("SGS_unpack3_1")
    unpack31_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack31_multiv2_tensor")
    unpack31_multiv2_out_tensors.append("unpack31_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack31_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack31_multiv2",unpack31_multiv2_in_tensors,unpack31_multiv2_out_tensors)

    unpack32_multiv2_out_tensors = []
    unpack32_multiv2_in_tensors = []
    unpack32_multiv2_in_tensors.append("SGS_unpack3_2")
    unpack32_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack32_multiv2_tensor")
    unpack32_multiv2_out_tensors.append("unpack32_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack32_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack32_multiv2",unpack32_multiv2_in_tensors,unpack32_multiv2_out_tensors)

    unpack33_multiv2_out_tensors = []
    unpack33_multiv2_in_tensors = []
    unpack33_multiv2_in_tensors.append("SGS_unpack3_3")
    unpack33_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack33_multiv2_tensor")
    unpack33_multiv2_out_tensors.append("unpack33_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack33_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack33_multiv2",unpack33_multiv2_in_tensors,unpack33_multiv2_out_tensors)

    unpack34_multiv2_out_tensors = []
    unpack34_multiv2_in_tensors = []
    unpack34_multiv2_in_tensors.append("SGS_unpack3_4")
    unpack34_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack34_multiv2_tensor")
    unpack34_multiv2_out_tensors.append("unpack34_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack34_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack34_multiv2",unpack34_multiv2_in_tensors,unpack34_multiv2_out_tensors)

    unpack35_multiv2_out_tensors = []
    unpack35_multiv2_in_tensors = []
    unpack35_multiv2_in_tensors.append("SGS_unpack3_5")
    unpack35_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack35_multiv2_tensor")
    unpack35_multiv2_out_tensors.append("unpack35_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack35_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack35_multiv2",unpack35_multiv2_in_tensors,unpack35_multiv2_out_tensors)

    unpack36_multiv2_out_tensors = []
    unpack36_multiv2_in_tensors = []
    unpack36_multiv2_in_tensors.append("SGS_unpack3_6")
    unpack36_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack36_multiv2_tensor")
    unpack36_multiv2_out_tensors.append("unpack36_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack36_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack36_multiv2",unpack36_multiv2_in_tensors,unpack36_multiv2_out_tensors)

    unpack37_multiv2_out_tensors = []
    unpack37_multiv2_in_tensors = []
    unpack37_multiv2_in_tensors.append("SGS_unpack3_7")
    unpack37_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack37_multiv2_tensor")
    unpack37_multiv2_out_tensors.append("unpack37_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack37_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack37_multiv2",unpack37_multiv2_in_tensors,unpack37_multiv2_out_tensors)

    unpack38_multiv2_out_tensors = []
    unpack38_multiv2_in_tensors = []
    unpack38_multiv2_in_tensors.append("SGS_unpack3_8")
    unpack38_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack38_multiv2_tensor")
    unpack38_multiv2_out_tensors.append("unpack38_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack38_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack38_multiv2",unpack38_multiv2_in_tensors,unpack38_multiv2_out_tensors)

    unpack39_multiv2_out_tensors = []
    unpack39_multiv2_in_tensors = []
    unpack39_multiv2_in_tensors.append("SGS_unpack3_9")
    unpack39_multiv2_in_tensors.append("variances2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack39_multiv2_tensor")
    unpack39_multiv2_out_tensors.append("unpack39_multiv2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack39_multiv2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack39_multiv2",unpack39_multiv2_in_tensors,unpack39_multiv2_out_tensors)


    unpack30_multiv2_mulanchor2_out_tensors = []
    unpack30_multiv2_mulanchor2_in_tensors = []
    unpack30_multiv2_mulanchor2_in_tensors.append("unpack30_multiv2_tensor")
    unpack30_multiv2_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack30_multiv2_mulanchor2_tensor")
    unpack30_multiv2_mulanchor2_out_tensors.append("unpack30_multiv2_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack30_multiv2_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack30_multiv2_mulanchor2",unpack30_multiv2_mulanchor2_in_tensors,unpack30_multiv2_mulanchor2_out_tensors)

    unpack31_multiv2_mulanchor3_out_tensors = []
    unpack31_multiv2_mulanchor3_in_tensors = []
    unpack31_multiv2_mulanchor3_in_tensors.append("unpack31_multiv2_tensor")
    unpack31_multiv2_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack31_multiv2_mulanchor3_tensor")
    unpack31_multiv2_mulanchor3_out_tensors.append("unpack31_multiv2_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack31_multiv2_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack31_multiv2_mulanchor3",unpack31_multiv2_mulanchor3_in_tensors,unpack31_multiv2_mulanchor3_out_tensors)

    unpack32_multiv2_mulanchor2_out_tensors = []
    unpack32_multiv2_mulanchor2_in_tensors = []
    unpack32_multiv2_mulanchor2_in_tensors.append("unpack32_multiv2_tensor")
    unpack32_multiv2_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack32_multiv2_mulanchor2_tensor")
    unpack32_multiv2_mulanchor2_out_tensors.append("unpack32_multiv2_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack32_multiv2_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack32_multiv2_mulanchor2",unpack32_multiv2_mulanchor2_in_tensors,unpack32_multiv2_mulanchor2_out_tensors)

    unpack33_multiv2_mulanchor3_out_tensors = []
    unpack33_multiv2_mulanchor3_in_tensors = []
    unpack33_multiv2_mulanchor3_in_tensors.append("unpack33_multiv2_tensor")
    unpack33_multiv2_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack33_multiv2_mulanchor3_tensor")
    unpack33_multiv2_mulanchor3_out_tensors.append("unpack33_multiv2_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack33_multiv2_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack33_multiv2_mulanchor3",unpack33_multiv2_mulanchor3_in_tensors,unpack33_multiv2_mulanchor3_out_tensors)

    unpack34_multiv2_mulanchor2_out_tensors = []
    unpack34_multiv2_mulanchor2_in_tensors = []
    unpack34_multiv2_mulanchor2_in_tensors.append("unpack34_multiv2_tensor")
    unpack34_multiv2_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack34_multiv2_mulanchor2_tensor")
    unpack34_multiv2_mulanchor2_out_tensors.append("unpack34_multiv2_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack34_multiv2_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack34_multiv2_mulanchor2",unpack34_multiv2_mulanchor2_in_tensors,unpack34_multiv2_mulanchor2_out_tensors)

    unpack35_multiv2_mulanchor3_out_tensors = []
    unpack35_multiv2_mulanchor3_in_tensors = []
    unpack35_multiv2_mulanchor3_in_tensors.append("unpack35_multiv2_tensor")
    unpack35_multiv2_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack35_multiv2_mulanchor3_tensor")
    unpack35_multiv2_mulanchor3_out_tensors.append("unpack35_multiv2_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack35_multiv2_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack35_multiv2_mulanchor3",unpack35_multiv2_mulanchor3_in_tensors,unpack35_multiv2_mulanchor3_out_tensors)

    unpack36_multiv2_mulanchor2_out_tensors = []
    unpack36_multiv2_mulanchor2_in_tensors = []
    unpack36_multiv2_mulanchor2_in_tensors.append("unpack36_multiv2_tensor")
    unpack36_multiv2_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack36_multiv2_mulanchor2_tensor")
    unpack36_multiv2_mulanchor2_out_tensors.append("unpack36_multiv2_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack36_multiv2_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack36_multiv2_mulanchor2",unpack36_multiv2_mulanchor2_in_tensors,unpack36_multiv2_mulanchor2_out_tensors)

    unpack37_multiv2_mulanchor3_out_tensors = []
    unpack37_multiv2_mulanchor3_in_tensors = []
    unpack37_multiv2_mulanchor3_in_tensors.append("unpack37_multiv2_tensor")
    unpack37_multiv2_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack37_multiv2_mulanchor3_tensor")
    unpack37_multiv2_mulanchor3_out_tensors.append("unpack37_multiv2_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack37_multiv2_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack37_multiv2_mulanchor3",unpack37_multiv2_mulanchor3_in_tensors,unpack37_multiv2_mulanchor3_out_tensors)

    unpack38_multiv2_mulanchor2_out_tensors = []
    unpack38_multiv2_mulanchor2_in_tensors = []
    unpack38_multiv2_mulanchor2_in_tensors.append("unpack38_multiv2_tensor")
    unpack38_multiv2_mulanchor2_in_tensors.append("anchor2_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack38_multiv2_mulanchor2_tensor")
    unpack38_multiv2_mulanchor2_out_tensors.append("unpack38_multiv2_mulanchor2_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack38_multiv2_mulanchor2",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack38_multiv2_mulanchor2",unpack38_multiv2_mulanchor2_in_tensors,unpack38_multiv2_mulanchor2_out_tensors)

    unpack39_multiv2_mulanchor3_out_tensors = []
    unpack39_multiv2_mulanchor3_in_tensors = []
    unpack39_multiv2_mulanchor3_in_tensors.append("unpack39_multiv2_tensor")
    unpack39_multiv2_mulanchor3_in_tensors.append("anchor3_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack39_multiv2_mulanchor3_tensor")
    unpack39_multiv2_mulanchor3_out_tensors.append("unpack39_multiv2_mulanchor3_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack39_multiv2_mulanchor3",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_unpack39_multiv2_mulanchor3",unpack39_multiv2_mulanchor3_in_tensors,unpack39_multiv2_mulanchor3_out_tensors)

    unpack30_multiv2_mulanchor2_addanchor4_out_tensors = []
    unpack30_multiv2_mulanchor2_addanchor4_in_tensors = []
    unpack30_multiv2_mulanchor2_addanchor4_in_tensors.append("unpack30_multiv2_mulanchor2_tensor")
    unpack30_multiv2_mulanchor2_addanchor4_in_tensors.append("anchor4_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack30_multiv2_mulanchor2_addanchor4_tensor")
    unpack30_multiv2_mulanchor2_addanchor4_out_tensors.append("unpack30_multiv2_mulanchor2_addanchor4_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack30_multiv2_mulanchor2_addanchor4",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack30_multiv2_mulanchor2_addanchor4",unpack30_multiv2_mulanchor2_addanchor4_in_tensors,unpack30_multiv2_mulanchor2_addanchor4_out_tensors)

    unpack31_multiv2_mulanchor3_addanchor5_out_tensors = []
    unpack31_multiv2_mulanchor3_addanchor5_in_tensors = []
    unpack31_multiv2_mulanchor3_addanchor5_in_tensors.append("unpack31_multiv2_mulanchor3_tensor")
    unpack31_multiv2_mulanchor3_addanchor5_in_tensors.append("anchor5_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack31_multiv2_mulanchor3_addanchor5_tensor")
    unpack31_multiv2_mulanchor3_addanchor5_out_tensors.append("unpack31_multiv2_mulanchor3_addanchor5_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack31_multiv2_mulanchor3_addanchor5",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack31_multiv2_mulanchor3_addanchor5",unpack31_multiv2_mulanchor3_addanchor5_in_tensors,unpack31_multiv2_mulanchor3_addanchor5_out_tensors)

    unpack32_multiv2_mulanchor2_addanchor6_out_tensors = []
    unpack32_multiv2_mulanchor2_addanchor6_in_tensors = []
    unpack32_multiv2_mulanchor2_addanchor6_in_tensors.append("unpack32_multiv2_mulanchor2_tensor")
    unpack32_multiv2_mulanchor2_addanchor6_in_tensors.append("anchor6_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack32_multiv2_mulanchor2_addanchor6_tensor")
    unpack32_multiv2_mulanchor2_addanchor6_out_tensors.append("unpack32_multiv2_mulanchor2_addanchor6_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack32_multiv2_mulanchor2_addanchor6",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack32_multiv2_mulanchor2_addanchor6",unpack32_multiv2_mulanchor2_addanchor6_in_tensors,unpack32_multiv2_mulanchor2_addanchor6_out_tensors)

    unpack33_multiv2_mulanchor3_addanchor7_out_tensors = []
    unpack33_multiv2_mulanchor3_addanchor7_in_tensors = []
    unpack33_multiv2_mulanchor3_addanchor7_in_tensors.append("unpack33_multiv2_mulanchor3_tensor")
    unpack33_multiv2_mulanchor3_addanchor7_in_tensors.append("anchor7_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack33_multiv2_mulanchor3_addanchor7_tensor")
    unpack33_multiv2_mulanchor3_addanchor7_out_tensors.append("unpack33_multiv2_mulanchor3_addanchor7_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack33_multiv2_mulanchor3_addanchor7",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack33_multiv2_mulanchor3_addanchor7",unpack33_multiv2_mulanchor3_addanchor7_in_tensors,unpack33_multiv2_mulanchor3_addanchor7_out_tensors)

    unpack34_multiv2_mulanchor2_addanchor8_out_tensors = []
    unpack34_multiv2_mulanchor2_addanchor8_in_tensors = []
    unpack34_multiv2_mulanchor2_addanchor8_in_tensors.append("unpack34_multiv2_mulanchor2_tensor")
    unpack34_multiv2_mulanchor2_addanchor8_in_tensors.append("anchor8_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack34_multiv2_mulanchor2_addanchor8_tensor")
    unpack34_multiv2_mulanchor2_addanchor8_out_tensors.append("unpack34_multiv2_mulanchor2_addanchor8_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack34_multiv2_mulanchor2_addanchor8",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack34_multiv2_mulanchor2_addanchor8",unpack34_multiv2_mulanchor2_addanchor8_in_tensors,unpack34_multiv2_mulanchor2_addanchor8_out_tensors)

    unpack35_multiv2_mulanchor3_addanchor9_out_tensors = []
    unpack35_multiv2_mulanchor3_addanchor9_in_tensors = []
    unpack35_multiv2_mulanchor3_addanchor9_in_tensors.append("unpack35_multiv2_mulanchor3_tensor")
    unpack35_multiv2_mulanchor3_addanchor9_in_tensors.append("anchor9_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack35_multiv2_mulanchor3_addanchor9_tensor")
    unpack35_multiv2_mulanchor3_addanchor9_out_tensors.append("unpack35_multiv2_mulanchor3_addanchor9_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack35_multiv2_mulanchor3_addanchor9",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack35_multiv2_mulanchor3_addanchor9",unpack35_multiv2_mulanchor3_addanchor9_in_tensors,unpack35_multiv2_mulanchor3_addanchor9_out_tensors)

    unpack36_multiv2_mulanchor2_addanchor10_out_tensors = []
    unpack36_multiv2_mulanchor2_addanchor10_in_tensors = []
    unpack36_multiv2_mulanchor2_addanchor10_in_tensors.append("unpack36_multiv2_mulanchor2_tensor")
    unpack36_multiv2_mulanchor2_addanchor10_in_tensors.append("anchor10_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack36_multiv2_mulanchor2_addanchor10_tensor")
    unpack36_multiv2_mulanchor2_addanchor10_out_tensors.append("unpack36_multiv2_mulanchor2_addanchor10_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack36_multiv2_mulanchor2_addanchor10",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack36_multiv2_mulanchor2_addanchor10",unpack36_multiv2_mulanchor2_addanchor10_in_tensors,unpack36_multiv2_mulanchor2_addanchor10_out_tensors)

    unpack37_multiv2_mulanchor3_addanchor11_out_tensors = []
    unpack37_multiv2_mulanchor3_addanchor11_in_tensors = []
    unpack37_multiv2_mulanchor3_addanchor11_in_tensors.append("unpack37_multiv2_mulanchor3_tensor")
    unpack37_multiv2_mulanchor3_addanchor11_in_tensors.append("anchor11_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack37_multiv2_mulanchor3_addanchor11_tensor")
    unpack37_multiv2_mulanchor3_addanchor11_out_tensors.append("unpack37_multiv2_mulanchor3_addanchor11_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack37_multiv2_mulanchor3_addanchor11",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack37_multiv2_mulanchor3_addanchor11",unpack37_multiv2_mulanchor3_addanchor11_in_tensors,unpack37_multiv2_mulanchor3_addanchor11_out_tensors)

    unpack38_multiv2_mulanchor2_addanchor12_out_tensors = []
    unpack38_multiv2_mulanchor2_addanchor12_in_tensors = []
    unpack38_multiv2_mulanchor2_addanchor12_in_tensors.append("unpack38_multiv2_mulanchor2_tensor")
    unpack38_multiv2_mulanchor2_addanchor12_in_tensors.append("anchor12_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack38_multiv2_mulanchor2_addanchor12_tensor")
    unpack38_multiv2_mulanchor2_addanchor12_out_tensors.append("unpack38_multiv2_mulanchor2_addanchor12_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack38_multiv2_mulanchor2_addanchor12",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack38_multiv2_mulanchor2_addanchor12",unpack38_multiv2_mulanchor2_addanchor12_in_tensors,unpack38_multiv2_mulanchor2_addanchor12_out_tensors)

    unpack39_multiv2_mulanchor3_addanchor13_out_tensors = []
    unpack39_multiv2_mulanchor3_addanchor13_in_tensors = []
    unpack39_multiv2_mulanchor3_addanchor13_in_tensors.append("unpack39_multiv2_mulanchor3_tensor")
    unpack39_multiv2_mulanchor3_addanchor13_in_tensors.append("anchor13_tensor")
    sgs_builder.buildTensor(model_config["shape"],"unpack39_multiv2_mulanchor3_addanchor13_tensor")
    unpack39_multiv2_mulanchor3_addanchor13_out_tensors.append("unpack39_multiv2_mulanchor3_addanchor13_tensor")
    sgs_builder.buildOperatorCode("SGS_unpack39_multiv2_mulanchor3_addanchor13",tflite.BuiltinOperator.BuiltinOperator().ADD)
    sgs_builder.buildOperator("SGS_unpack39_multiv2_mulanchor3_addanchor13",unpack39_multiv2_mulanchor3_addanchor13_in_tensors,unpack39_multiv2_mulanchor3_addanchor13_out_tensors)


    concat_in_tensors = []

    concat_in_tensors.append("unpack30_multiv2_mulanchor2_addanchor4_tensor")
    concat_in_tensors.append("unpack31_multiv2_mulanchor3_addanchor5_tensor")
    concat_in_tensors.append("unpack32_multiv2_mulanchor2_addanchor6_tensor")
    concat_in_tensors.append("unpack33_multiv2_mulanchor3_addanchor7_tensor")
    concat_in_tensors.append("unpack34_multiv2_mulanchor2_addanchor8_tensor")
    concat_in_tensors.append("unpack35_multiv2_mulanchor3_addanchor9_tensor")
    concat_in_tensors.append("unpack36_multiv2_mulanchor2_addanchor10_tensor")
    concat_in_tensors.append("unpack37_multiv2_mulanchor3_addanchor11_tensor")
    concat_in_tensors.append("unpack38_multiv2_mulanchor2_addanchor12_tensor")
    concat_in_tensors.append("unpack39_multiv2_mulanchor3_addanchor13_tensor")
    concat_out_tensors = []
    sgs_builder.buildTensor([10,10140],"lms_concat")
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
    concat_transpose_out_shape =  [10140,10]
    concat_transpose_out_tensors = []
    sgs_builder.buildTensor(concat_transpose_out_shape,"concat_transpose_tensor")
    concat_transpose_out_tensors.append("concat_transpose_tensor")
    sgs_builder.buildOperatorCode("SGS_concat_transpose",tflite.BuiltinOperator.BuiltinOperator().TRANSPOSE)
    sgs_builder.buildOperator("SGS_concat_transpose",concat_transpose_in_tensors, concat_transpose_out_tensors)


    """=========================================="""

    nms_out_tensors = []
    nms_in_tensors = []
    nms_in_tensors.append("y1_tensor")
    nms_in_tensors.append("y2_tensor")
    nms_in_tensors.append("x1_tensor")
    nms_in_tensors.append("x2_tensor")
    nms_in_tensors.append("unpack2sub_logistic_tensor")

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
    1.标识输入tensor 和 coordinate score confidence class facecoordinate 对应关系
    2.nms 计算方式 fast nms / yolo_nms
    3.背景是否参与运算
    4.标识输出tensor 和  bboxcount bboxcoordinate score class facecoordinate 对应关系
    5.是否需要将input coordinate clip 到[0,1]
                       (b"input_class_idx",4,"int"),
                   (b"input_score_idx",5,"int"),
                   (b"input_confidence_idx",6,"int"),
    '''

    cus_options = [(b"input_coordinate_x1",2,"int"),
                   (b"input_coordinate_y1",0,"int"),
                   (b"input_coordinate_x2",3,"int"),
                   (b"input_coordinate_y2",1,"int"),
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
                   (b"max_detections",MAX_DETECTIONS,"int"),
                   (b"max_classes_per_detection",1,"int"),
                   (b"detections_per_class",1,"int"),
                   (b"num_classes",1,"int"),
                   (b"bmax_score",1,"int"),
                   (b"offline",0,"int"),
                   (b"num_classes_with_background",1,"int"),
                   (b"nms_score_threshold",0.200000003,"float"),
                   (b"nms_iou_threshold",0.300000012,"float")]
    options = sgs_builder.createFlexBuffer(cus_options)
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
    if getIPUVersion() == 'M6' or getIPUVersion() == 'I6E':
        file_identifier = b'TFL3'
    else:
        file_identifier = b'SIM2'
    sgs_builder.builder.Finish(sgs_builder.model, file_identifier)
    buf = sgs_builder.builder.Output()
    return buf

def anchors_set(cx,cy,w,h,fda_anchors14):
    cy = min(cy, 1.0)
    cx = min(cx, 1.0)
    w = min(w, 1.0)
    h = min(h, 1.0)

    fda_anchors14.append(cx)
    fda_anchors14.append(cy)
    fda_anchors14.append(w)
    fda_anchors14.append(h)
    fda_anchors14.append(cx - w * 0.25)
    fda_anchors14.append(cy - h * 0.25)
    fda_anchors14.append(cx + w * 0.25)
    fda_anchors14.append(cy - h * 0.25)
    fda_anchors14.append(cx)
    fda_anchors14.append(cy)
    fda_anchors14.append(cx - w * 0.2)
    fda_anchors14.append(cy + h * 0.2)
    fda_anchors14.append(cx + w * 0.2)
    fda_anchors14.append(cy + h * 0.2)


def anchors_generate(h, w, fda_anchors14):
    sgs_feature_map_sizes = anchor.get_anchors_num(h, w)
    anchors_num = sgs_feature_map_sizes[0]*sgs_feature_map_sizes[1]*21+sgs_feature_map_sizes[2]*sgs_feature_map_sizes[3]*1+sgs_feature_map_sizes[4]*sgs_feature_map_sizes[5]*1
    steps = [32.0, 64.0, 128.0]
    scales = [32.0, 64.0, 128.0, 256.0, 512.0]
    for y in range(sgs_feature_map_sizes[0].astype(np.int32)):
        for x in range(sgs_feature_map_sizes[1].astype(np.int32)):
            c_y = y + 0.5
            c_x = x + 0.5
            for y_i in range(-375, 375+1, 250):
                y_i = y_i/1000
                c_y_32 = (c_y + y_i) * steps[0];
                for x_i in range(-375, 375+1, 250):
                    x_i = x_i/1000
                    c_x_32 = ((c_x + x_i) * steps[0])
                    c_w = scales[0]
                    c_h = scales[0]
                    anchors_set(c_x_32/w, c_y_32/h, c_w/w, c_h/h, fda_anchors14)
            for y_i in range(-25, 25+1, 50):
                y_i = y_i/100
                c_y_64 = (c_y + y_i) * steps[0]
                for x_i in range(-25, 25+1, 50):
                    x_i = x_i/100
                    c_x_64 = (c_x + x_i) * steps[0]
                    c_w = scales[1]
                    c_h = scales[1]
                    anchors_set(c_x_64/w, c_y_64/h, c_w/w, c_h/h, fda_anchors14)
            c_x_128 = (c_x + 0) * steps[0]
            c_y_128 = (c_y + 0) * steps[0]
            c_w = scales[2]
            c_h = scales[2]
            anchors_set(c_x_128/w, c_y_128/h, c_w/w, c_h/h, fda_anchors14)
    for y in range(sgs_feature_map_sizes[2].astype(np.int32)):
        c_y = (y + 0.5) * steps[1]
        for x in range(sgs_feature_map_sizes[3].astype(np.int32)):
            c_x = (x + 0.5) * steps[1]
            c_w = scales[3]
            c_h = scales[3]
            anchors_set(c_x/w, c_y/h, c_w/w, c_h/h, fda_anchors14)
    for y in range(sgs_feature_map_sizes[4].astype(np.int32)):
        c_y = (y + 0.5) * steps[2]
        for x in range(sgs_feature_map_sizes[5].astype(np.int32)):
            c_x = (x + 0.5) * steps[2]
            c_w = scales[4]
            c_h = scales[4]
            anchors_set(c_x/w, c_y/h, c_w/w, c_h/h, fda_anchors14)


def get_postprocess():
    model_config = {"name":"caffe_fda_800x600",
          "input" : ['283','276','290'],
          "input_shape" : [[1,1,4,10140],[1,1,2,10140],[1,1,10,10140]],
          "shape" : [1,10140],
          "out_shapes" : [[1,MAX_DETECTIONS,4],[1,MAX_DETECTIONS],[1,MAX_DETECTIONS],[1],[1,MAX_DETECTIONS],[1,10140,10]],
          "input_hw": [600., 800.]}

    fda = TFLitePostProcess()
    fda_buf = buildGraph(fda,model_config)
    outfilename = model_config["name"] + "_postprocess.sim"
    with open(outfilename, 'wb') as f:
        f.write(fda_buf)
        f.close()
    print("\nWell Done!" + outfilename  + " generated!\n")
    return outfilename

def model_postprocess():
    return get_postprocess()
