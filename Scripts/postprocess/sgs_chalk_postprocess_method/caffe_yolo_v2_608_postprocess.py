import math
import os
import pdb
import numpy as np
import sys
import six
import calibrator_custom
from calibrator_custom import sgs_chalk
from third_party import tflite

MAX_DETECTIONS = 100
if calibrator_custom.utils.get_sdk_version() in ['L10']:
    MAX_DETECTIONS = 64

def buildGraph(model_config):
    """

    :return:
    """

    box_num = 5
    side_x = [19]
    side_y = [19]
    num_classes = 80
    num_anchors = 0
    for i in range(len(side_x)):
        num_anchors += box_num * side_x[i] * side_y[i]
    ppw = np.ones(num_anchors)
    pph = np.ones(num_anchors)

    py=[]
    px=[]
    for k in range(len(side_x)):
        for i in range(side_y[k]*side_x[k]):
            tempX = i%side_y[k]
            tempY = math.floor(i/side_y[k])
            for n in range(box_num):
                px.append(tempX)
                py.append(tempY)

    pw = np.ones(num_anchors)
    ph = np.ones(num_anchors)

    sx = np.ones(num_anchors)*(1.0/side_y[0])
    sy = np.ones(num_anchors)*(1.0/side_x[0])

    biases= [[0.57273, 0.677385],[1.87446, 2.06253],[3.33843, 5.47434],[7.88282, 3.52778],[9.77052, 9.16828]]
    sw = [x[0]/(2*side_x[0]) for x in biases ]*(side_x[0]*side_y[0])
    sh = [x[1]/(2*side_y[0]) for x in biases ]*(side_x[0]*side_y[0])

    config = {"shape" : [1,num_anchors],
          "tx_func" : (tflite.BuiltinOperator.BuiltinOperator().LOGISTIC,None),#None or 'x_scale'
          "ty_func" : (tflite.BuiltinOperator.BuiltinOperator().LOGISTIC,None),#None or 'y_scale'
          "tw_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'w_scale'
          "th_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'h_scale'
          "x_scale" : 0.1,
          "y_scale" : 0.1,
          "w_scale" : 1,
          "h_scale" : 1,
          "anchor_selector" : "constant",
          "pw" : pw,
          "ph" : ph,
          "pw_func" : (None,None),
          "ph_func" : (None,None),
          "ppw" : ppw,
          "px" : px,
          "pph" : pph,
          "py" : py,
          "sx" : sx,
          "sy" : sy,
          "sw" : sw,
          "sh" : sh
          }

    in0 = sgs_chalk.Input(model_config["input_shape"], name = model_config["input"][0])
    reshape_tensor = sgs_chalk.Reshape(in0,[1,side_x[0]*side_y[0]*box_num,num_classes+5],name = "reshape_tensor")

    unpack_out_tensors1 = []
    for i in range(7):
        unpack_out_tensors1.append("SGS_unpack"+str(i))
    output_list = sgs_chalk.PostProcess_Unpack(reshape_tensor,num_classes=80,name=unpack_out_tensors1)

    bosdecoder_output_list = sgs_chalk.BoxDecoder(config,output_list)
    confidence_tensor = sgs_chalk.Logistic(output_list[4],name="confidence_tensor")
    SGS_score0 = sgs_chalk.Logistic(output_list[5],name="score0_tensor")
    SGS_score1 = sgs_chalk.Mul(confidence_tensor,SGS_score0,name="SGS_score1")

    """=========================================="""
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
    out1 = sgs_chalk.TFLite_Detection_NMS(bosdecoder_output_list[0],bosdecoder_output_list[1],bosdecoder_output_list[2],
                                          bosdecoder_output_list[3],confidence_tensor,SGS_score1,output_list[6],mode='YOLO',
                                          max_detections=MAX_DETECTIONS,nms_score_threshold=0.4,
                                          nms_iou_threshold=0.45,num_classes=80,is_need_index=False)

    """=========================================="""

    outfilename = model_config["name"] + "_postprocess.sim"
    model = sgs_chalk.Model([in0],out1,model_config["name"])
    model.save(outfilename)

    print("\nWell Done! " + outfilename  + " generated!\n")
    return outfilename


def get_postprocess():
    model_config = {"name":"caffe_yolo_v2_608",
          "input" : ['layer31-conv'],
          "input_shape" : [1,19,19,425]}

    outfilename = buildGraph(model_config)

    return outfilename

def model_postprocess():
    return get_postprocess()
