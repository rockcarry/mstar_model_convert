import math
import os
import pdb
import numpy as np
import sys
import six

from calibrator_custom import sgs_chalk

if 'IPU_TOOL' in os.environ:
    Project_path = os.environ['IPU_TOOL']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/postprocess"))
    anchors = np.load(os.path.join(Project_path, "Scripts/postprocess/", "anchors_ssd_tf.npy"))
elif 'SGS_IPU_DIR' in os.environ:
    Project_path = os.environ['SGS_IPU_DIR']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/postprocess"))
    anchors = np.load(os.path.join(Project_path, "Scripts/postprocess/", "anchors_ssd_tf.npy"))
else:
    raise OSError('Run `source cfg_env.sh` in top directory.')

from  anchor_param import *
from third_party import tflite



def buildGraph(model_config):
    """

    :return:
    """

    anchors0 = [x[0] for x in anchors]
    anchors1 = [x[1] for x in anchors]
    anchors2 = [x[2] for x in anchors]
    anchors3 = [x[3] for x in anchors]
    #pdb.set_trace()
    ppw = anchors2
    px = anchors0
    pph = anchors3
    py = anchors1
    pw = anchors2
    ph = anchors3

    sx = np.ones(1917)
    sy = np.ones(1917)
    sw = np.ones(1917)*0.5
    sh = np.ones(1917)*0.5

    config = {"shape" : [1,1917],
          "tx_func" : (tflite.BuiltinOperator.BuiltinOperator().MUL,"x_scale"),#None or 'x_scale'
          "ty_func" : (tflite.BuiltinOperator.BuiltinOperator().MUL,"y_scale"),#None or 'y_scale'
          "tw_func" : (tflite.BuiltinOperator.BuiltinOperator().MUL,"w_scale"),#None or 'w_scale'
          "th_func" : (tflite.BuiltinOperator.BuiltinOperator().MUL,"h_scale"),#None or 'h_scale'
          "x_scale" : 0.1,
          "y_scale" : 0.1,
          "w_scale" : 0.2,
          "h_scale" : 0.2,
          "anchor_selector" : "constant",
          "pw" : pw,
          "ph" : ph,
          "pw_func" : (tflite.BuiltinOperator.BuiltinOperator().SQUARE,None),
          "ph_func" : (tflite.BuiltinOperator.BuiltinOperator().SQUARE,None),
          "ppw" : ppw,
          "px" : px,
          "pph" : pph,
          "py" : py,
          "sx" : sx,
          "sy" : sy,
          "sw" : sw,
          "sh" : sh,
          }

    in0 = sgs_chalk.Input(model_config["input_shape"][0], name = model_config["input"][0])
    unpack_out_tensors1 = []
    for i in range(4):
        unpack_out_tensors1.append("SGS_unpack"+str(i))
    output_list = sgs_chalk.Unpack(in0,num=4,axis=2,name=unpack_out_tensors1)

    bosdecoder_output_list = sgs_chalk.BoxDecoder(config,output_list)

    convert_scores = sgs_chalk.Input(model_config["input_shape"][1], name = model_config["input"][1])

    postprocess_max_output_list = sgs_chalk.PostProcess_Max(convert_scores,num_classes=90,skip=1)
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
    out1 = sgs_chalk.TFLite_Detection_NMS(bosdecoder_output_list[1],bosdecoder_output_list[0],bosdecoder_output_list[3],
                                          bosdecoder_output_list[2],postprocess_max_output_list[0],postprocess_max_output_list[1],
                                          mode='SSD',max_detections=10,nms_score_threshold=9.99999994e-09,
                                          nms_iou_threshold=0.600000024,num_classes=90,is_need_index=True)


    """=========================================="""

    outfilename = model_config["name"] + "_postprocess.sim"
    model = sgs_chalk.Model([in0,convert_scores],out1,model_config["name"])
    model.save(outfilename)

    print("\nWell Done! " + outfilename  + " generated!\n")
    return outfilename


def get_postprocess():
    model_config = {"name":"ssd_mobilenet_v1",
          "input" : ["Squeeze","convert_scores"],
          "input_shape" : [[1,1917,4],[1,1917,91]],
          "shape" : [1,1917],
          "out_shapes" : [[1,10,4],[1,10],[1,10],[1],[1,10]]}

    outfilename = buildGraph(model_config)

    return outfilename

def model_postprocess():
    return get_postprocess()
